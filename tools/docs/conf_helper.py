class AutosummaryFilenameMap(dict):
    def get(self, k, default=None):
        return k.split('.')[-1]
