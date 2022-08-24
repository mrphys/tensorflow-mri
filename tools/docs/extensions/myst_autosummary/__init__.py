"""MyST-compatible drop-in replacement for Sphinx's Autosummary extension."""
__version__ = '0.1.0'

# This extension only overrides the Autosummary method that creates the
# summary table. The changes relative to the original code are minimal.
# Though it is possible some reST-specific content generation was
# overlooked elsewhere in Autosummary's code base. The stub generation
# was ignored. We would have to create .md files instead of .rst.

import os
import posixpath
import re

import docutils
from docutils import nodes

import sphinx
from sphinx import addnodes
from sphinx.ext.autodoc.directive import DocumenterBridge, Options
from sphinx.ext import autosummary
from sphinx.ext.autodoc.mock import mock
from sphinx.locale import __
from sphinx.util.matching import Matcher

logger = sphinx.util.logging.getLogger(__name__)


class autosummary_toc(nodes.comment):
    pass


class Autosummary(autosummary.Autosummary):
    """Extends the `autosummary` directive provided by Autosummary."""

    def run(self):
        """Reimplements the run method of the parent class.

        Only one line has been changed with respect to the parent class,
        indicated below.
        """
        self.bridge = DocumenterBridge(self.env, self.state.document.reporter,
                                       Options(), self.lineno, self.state)

        names = [x.strip().split()[0] for x in self.content
                 if x.strip() and re.search(r'^[~a-zA-Z_]', x.strip()[0])]
        items = self.get_items(names)
        nodes = self.get_table(items)

        if 'toctree' in self.options:
            dirname = posixpath.dirname(self.env.docname)

            tree_prefix = self.options['toctree'].strip()
            docnames = []
            excluded = Matcher(self.config.exclude_patterns)
            filename_map = self.config.autosummary_filename_map
            for _name, _sig, _summary, real_name in items:
                real_name = filename_map.get(real_name, real_name)
                docname = posixpath.join(tree_prefix, real_name)
                docname = posixpath.normpath(posixpath.join(dirname, docname))
                if docname not in self.env.found_docs:
                    if excluded(self.env.doc2path(docname, False)):
                        msg = __('autosummary references excluded document %r. Ignored.')
                    else:
                        msg = __('autosummary: stub file not found %r. '
                                 'Check your autosummary_generate setting.')

                    logger.warning(msg, real_name, location=self.get_location())
                    continue

                docnames.append(docname)

            if docnames:
                tocnode = addnodes.toctree()
                tocnode['includefiles'] = docnames
                # This is the only line that is different from the parent class.
                # This makes for cleaner TOC entries.
                tocnode['entries'] = [(docn.split('/')[-1], docn) for docn in docnames]
                tocnode['maxdepth'] = -1
                tocnode['glob'] = None
                tocnode['caption'] = self.options.get('caption')

                nodes.append(autosummary_toc('', '', tocnode))

        if 'toctree' not in self.options and 'caption' in self.options:
            logger.warning(__('A captioned autosummary requires :toctree: option. ignored.'),
                           location=nodes[-1])

        return nodes


    def get_table(self, items):
        """
        Reimplements the generation of the summary table.

        This new method returns Docutils nodes containing MyST-style
        object references instead of standard Sphinx roles. It simply
        regenerates the content. (It may also be possible to call the
        method of the parent class and convert the syntax with a
        regular expression after it's been generated.)
        """
        table_spec = sphinx.addnodes.tabular_col_spec()
        table_spec['spec'] = r'\X{1}{2}\X{1}{2}'

        table = autosummary.autosummary_table('')
        real_table = docutils.nodes.table('', classes=['longtable'])
        table.append(real_table)
        group = docutils.nodes.tgroup('', cols=2)
        real_table.append(group)
        group.append(docutils.nodes.colspec('', colwidth=10))
        group.append(docutils.nodes.colspec('', colwidth=90))
        body = docutils.nodes.tbody('')
        group.append(body)

        def append_row(*column_texts: str) -> None:
            row = docutils.nodes.row('')
            (source, line) = self.state_machine.get_source_and_line()
            for text in column_texts:
                node = docutils.nodes.paragraph('')
                vl = docutils.statemachine.StringList()
                vl.append(text, f'{source}:{line:d}:<autosummary>')
                with sphinx.util.docutils.switch_source_input(self.state, vl):
                    self.state.nested_parse(vl, 0, node)
                    try:
                        if isinstance(node[0], docutils.nodes.paragraph):
                            node = node[0]
                    except IndexError:
                        pass
                    row.append(docutils.nodes.entry('', node))
            body.append(row)

        for (name, sig, summary, real_name) in items:
            if 'nosignatures' not in self.options:
                item = ('{py:obj}' + f'`{name} <{real_name}>`\\ ' +
                        sphinx.util.rst.escape(sig))
            else:
                item = '{py:obj}' + f'`{name} <{real_name}>`'
            append_row(item, summary)

        return [table_spec, table]


def get_md_suffix(app):
    """Replaces `get_rst_suffix` in original `autosummary` extension."""
    return '.md'


def process_generate_options(app):
    genfiles = app.config.autosummary_generate

    if genfiles is True:
        env = app.builder.env
        genfiles = [env.doc2path(x, base=None) for x in env.found_docs
                    if os.path.isfile(env.doc2path(x))]
    elif genfiles is False:
        pass
    else:
        ext = list(app.config.source_suffix)
        genfiles = [genfile + (ext[0] if not genfile.endswith(tuple(ext)) else '')
                    for genfile in genfiles]

        for entry in genfiles[:]:
            if not os.path.isfile(os.path.join(app.srcdir, entry)):
                logger.warning(__('autosummary_generate: file not found: %s'), entry)
                genfiles.remove(entry)

    if not genfiles:
        return

    suffix = get_md_suffix(app)

    if suffix is None:
        logger.warning(__('autosummary generats .rst files internally. '
                          'But your source_suffix does not contain .rst. Skipped.'))
        return

    from extensions.myst_autosummary.generate import generate_autosummary_docs

    imported_members = app.config.autosummary_imported_members
    with mock(app.config.autosummary_mock_imports):
        generate_autosummary_docs(genfiles, suffix=suffix, base_path=app.srcdir,
                                  app=app, imported_members=imported_members,
                                  overwrite=app.config.autosummary_generate_overwrite,
                                  encoding=app.config.source_encoding)


def setup(app):
    """
    Sets up the extension.

    Sphinx calls this function if the user named the extension in `conf.py`.
    It then sets up the Autosummary extension that ships with Sphinx  and
    overrides whatever necessary to produce Markdown to be parsed by MyST
    instead of reStructuredText parsed by Sphinx/Docutils.
    """
    app.setup_extension('sphinx.ext.autosummary')
    app.add_directive('autosummary', Autosummary, override=True)
    app.connect('builder-inited', process_generate_options)
    return {'version': __version__, 'parallel_read_safe': True}
