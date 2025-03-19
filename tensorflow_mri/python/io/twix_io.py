# Copyright 2022 University College London. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""I/O operations with TWIX RAID files (Siemens raw data)."""

import copy
import dataclasses
import inspect
import io
import itertools
import os
import re
import reprlib
import struct
import sys
import typing
import warnings

import numpy as np
import tensorflow as tf

from tensorflow_mri.python.util import api_util

NUL = b'\x00'


@api_util.export("io.parse_twix")
def parse_twix(contents):
  """Parses the contents of a TWIX RAID file (Siemens raw data).

  .. warning::
    This function does not support graph execution.

  Example:
    >>> # Read bytes from file.
    >>> contents = tf.io.read_file("/path/to/file.dat")
    >>> # Parse the contents.
    >>> twix = tfmri.io.parse_twix(contents)
    >>> # Access the first measurement.
    >>> meas = twix.measurements[0]
    >>> # Get the protocol...
    >>> protocol = meas.protocol
    >>> # You can index the protocol to access any of the protocol buffers,
    >>> # e.g., the measurement protocol.
    >>> meas_prot = protocol['Meas']
    >>> # Protocol buffers are nested structures accessible with "dot notation"
    >>> # or "bracket notation". The following are equivalent:
    >>> base_res = meas_prot.MEAS.sKSpace.lBaseResolution.value
    >>> base_res = meas_prot['MEAS']['sKSpace']['lBaseResolution'].value
    >>> # The measurement object also contains the scan data.
    >>> scans = meas.scans
    >>> # Each scan has a header and the list of channels.
    >>> scan_header = scans[0].header
    >>> channels = scans[0].channels
    >>> # Each channel also has its own header as well as the raw measurement
    >>> # data.
    >>> channel_header = channels[0].header
    >>> data = channels[0].data

  Args:
    contents: A scalar `tf.Tensor` of type `string`. The encoded contents of a
      TWIX RAID file.

  Returns:
    A `TwixRaidFile` object containing the measurements, including protocol
    buffers, scan headers and scan data.

  Raises:
    RuntimeError: If called with eager execution disabled.
  """
  if not tf.executing_eagerly():
    raise RuntimeError('`parse_twix` must be called eagerly.')

  # To scalar array.
  if isinstance(contents, tf.Tensor):
    contents = contents.numpy()

  # Create a bytes stream.
  stream = io.BytesIO(contents)

  # Parse the object.
  twix = TwixRaidFile.parse(stream)

  # Close the stream. # TODO(jmontalt): This causes an error, so temporarily
  # disabled.
  # stream.close()

  return twix


@dataclasses.dataclass
class ChannelHeader():
  """Data structure for channel header."""
  # `type` and `channel_length` are unpacked from ulTypeAndChannelLength.
  # type_and_channel_length: int  # uint32
  type: int  # uint8
  channel_length:  int # uint24
  meas_uid: int  # int32
  scan_counter: int  # uint32
  reserved1: int  # uint32
  sequence_time: int  # uint32
  unused2: int  # uint32
  channel_id: int  # uint16
  unused3: int  # uint16
  crc: int  # uint32

  @classmethod
  def parse(cls, stream):
    fmt = '<IiIIIIHHI'
    values = struct.unpack(fmt, stream.read(struct.calcsize(fmt)))
    type_and_channel_length = values[0]
    type = type_and_channel_length & 0x000000ff  # pylint: disable=redefined-builtin
    channel_length = type_and_channel_length >> 8
    return cls(type, channel_length, *values[1:])

CHANNEL_HEADER_SIZE = 32


@dataclasses.dataclass
class ChannelData():
  """Data structure for channel data."""
  header: ChannelHeader
  data: tf.Tensor

  @classmethod
  def parse(cls, stream):
    header = ChannelHeader.parse(stream)
    return cls(header=header,
               data=_read_tensor_data(
                  stream, np.complex64,
                  header.channel_length - CHANNEL_HEADER_SIZE))


@dataclasses.dataclass
class SlicePosVec():
  """Data structure for slice position vector."""
  sag: float  # float32
  cor: float  # float32
  tra: float  # float32

  @classmethod
  def parse(cls, stream):
    return cls(*_read_float32(stream, 3))


@dataclasses.dataclass
class SliceData():
  """Data structure for slice data."""
  slice_pos_vec: SlicePosVec
  quaternion: typing.Tuple[float, float, float, float]  # float32 * 4

  @classmethod
  def parse(cls, stream):
    return cls(slice_pos_vec=SlicePosVec.parse(stream),
               quaternion=struct.unpack('<4f', stream.read(16)))


@dataclasses.dataclass
class CutOffData():
  """Data structure for cut-off data."""
  pre: int    # uint16
  post: int   # uint16

  @classmethod
  def parse(cls, stream):
    return cls(*_read_uint16(stream, 2))


@dataclasses.dataclass
class LoopCounters():
  """Data structure for loop counters."""
  line: int  # uint16 (same for all counters)
  acquisition: int
  slice: int
  partition: int
  echo: int
  phase: int
  repetition: int
  set: int
  seg: int
  ida: int
  idb: int
  idc: int
  idd: int
  ide: int

  @classmethod
  def parse(cls, stream):
    return cls(*_read_uint16(stream, 14))


@dataclasses.dataclass
class EvalInfoMask():
  """Data structure for evaluation info mask."""
  # pylint: disable=invalid-name
  ACQEND: bool  # 0: Last scan.
  RTFEEDBACK: bool  # 1: Realtime feedback scan.
  HPFEEDBACK: bool  # 2: High performance feedback scan.
  ONLINE: bool  # 3: Processing should be done online.
  OFFLINE: bool  # 4: Processing should be done offline.
  SYNCDATA: bool  # 5: Readout contains synchronous data.
  noname6: bool
  noname7: bool
  LASTSCANINCONCAT: bool  # 8: Last scan in concatenation.
  noname9: bool
  RAWDATACORRECTION: bool  # 10: Raw data correction scan.
  LASTSCANINMEAS: bool  # 11: Last scan in measurement.
  SCANSCALEFACTOR: bool  # 12: Scan-specific additional scale factor.
  _2NDHADAMARPULSE: bool  # 13: 2nd RF excitation of HADAMAR
  REFPHASESTABSCAN: bool  # 14: Reference phase stabilization scan.
  PHASESTABSCAN: bool  # 15: Phase stabilization scan.
  D3FFT: bool  # 16: Execute 3D FFT.
  SIGNREV: bool  # 17: Sign reversal.
  PHASEFFT: bool  # 18: Execute phase FFT.
  SWAPPED: bool  # 19: Swapped phase/readout direction.
  POSTSHAREDLINE: bool  # 20: Shared line.
  PHASCOR: bool  # 21: Phase correction data.
  PATREFSCAN: bool  # 22: Additional scan for PAT reference line/partition.
  PATREFANDIMASCAN: bool  # 23: PAT reference and image scan.
  REFLECT: bool   # 24: Reflect line.
  NOISEADJSCAN: bool  # 25: Noise adjustment scan.
  SHARENOW: bool  # 26: All lines acquired from the actual and previous.
  LASTMEASUREDLINE: bool  # 27: Current line is the last measured line.
  FIRSTSCANINSLICE: bool  # 28: First scan in slice (needed for time stamps).
  LASTSCANINSLICE: bool  # 29: Last scan in slice (needed for time stamps).
  TREFFECTIVEBEGIN: bool  # 30: Begin time stamp for TReff (triggered meas).
  TREFFECTIVEEND: bool  # 31: End time stamp for TReff (triggered meas).
  REF_POSITION: bool  # 32: Reference position for move during scan images.
  SLC_AVERAGED: bool  # 33: Averaged slice for slice partial averaging scheme.
  TAGFLAG1: bool  # 34: Adjust scan.
  CT_NORMALIZE: bool  # 35: Scan for correction maps for TimCT-Prescan norm.
  SCAN_FIRST: bool  # 36: First scan of a particular map.
  SCAN_LAST: bool  # 37: Last scan of a particular map.
  SLICE_ACCEL_REFSCAN: bool  # 38: Single-band refscan slice acc multiband acq.
  SLICE_ACCEL_PHASCOR: bool  # 39: Additional phase corr for slice acc.
  FIRST_SCAN_IN_BLADE: bool  # 40: First scan in blade.
  LAST_SCAN_IN_BLADE: bool  # 41: Last scan in blade.
  LAST_BLADE_IN_TR: bool  # 42: Last blade in TR interval.
  noname43: bool
  PACE: bool  # 44: PACE scan.
  RETRO_LASTPHASE: bool  # 45: Last phase in heartbeat.
  RETRO_ENDOFMEAS: bool  # 46: ADC at end of measurement.
  RETRO_REPEATTHISHEARTBEAT: bool  # 47: Triggers repeat of current heartbeat.
  RETRO_REPEATPREVHEARTBEAT: bool  # 48: Triggers repeat of previous heartbeat.
  RETRO_ABORTSCANNOW: bool  # 49: Just abort everything.
  RETRO_LASTHEARTBEAT: bool  # 50: This ADC is from last heart beat (dummy).
  RETRO_DUMMYSCAN: bool  # 51: Just a dummy scan, throw away.
  RETRO_ARRDETDISABLED: bool  # 52: Disables arrhythmia detection.
  B1_CONTROLLOOP: bool  # 53: Readout to be used for B1 control loop.
  SKIP_ONLINE_PHASCOR: bool  # 54: Skip online phase correction even if on.
  SKIP_REGRIDDING: bool  # 55: Skip regridding even if on.
  noname56: bool
  noname57: bool
  noname58: bool
  noname59: bool
  noname60: bool
  WIP_1: bool  # 61: Scan for WIP application "type 1".
  WIP_2: bool  # 62: Scan for WIP application "type 2".
  WIP_3: bool  # 63: Scan for WIP application "type 3".

  @classmethod
  def parse(cls, stream):
    mask = _read_uint64(stream)
    num_flags = 64
    bit_list = [bool((mask >> shift) & 1) for shift in range(num_flags)]
    return cls(*bit_list)


@dataclasses.dataclass
class ScanHeader():
  """Data structure for scan header."""
  dma_length: int  # uint25
  pack_bit: bool  # bool
  flags: int  # uint32
  meas_uid: int  # int32
  scan_counter: int  # uint32
  time_stamp: int  # uint32
  pmu_time_stamp: int  # uint32
  system_type: int  # uint16
  ptab_pos_delay: int  # uint16
  ptab_pos_x: int  # int32
  ptab_pos_y: int  # int32
  ptab_pos_z: int  # int32
  reserved1: int  # uint32
  eval_info_mask: EvalInfoMask  # uint64
  samples_in_scan: int  # uint16
  used_channels: int  # uint16
  loop_counters: LoopCounters
  cut_off: CutOffData
  kspace_centre_column: int  # uint16
  coil_select: int  # uint16
  read_out_offcentre: float  # float32
  time_since_last_rf: int  # uint32
  kspace_centre_line_no: int  # uint16
  kspace_centre_partition_no: int  # uint16
  slice_data: SliceData
  ice_program_para: typing.Tuple[int, ...]  # uint16 * 4
  reserved_para: typing.Tuple[int, ...]  # uint16 * 4
  application_counter: int  # uint16
  application_mask: int  # uint16
  crc: int  # uint32

  @classmethod
  def parse(cls, stream):
    flags_and_dma_length = _read_uint32(stream)
    return cls(dma_length=flags_and_dma_length % (2 ** 25),
               pack_bit = bool((flags_and_dma_length >> 25) & 1),
               flags = flags_and_dma_length >> 26,
               meas_uid=_read_int32(stream),
               scan_counter=_read_uint32(stream),
               time_stamp=_read_uint32(stream),
               pmu_time_stamp=_read_uint32(stream),
               system_type=_read_uint16(stream),
               ptab_pos_delay=_read_uint16(stream),
               ptab_pos_x=_read_int32(stream),
               ptab_pos_y=_read_int32(stream),
               ptab_pos_z=_read_int32(stream),
               reserved1=_read_uint32(stream),
               eval_info_mask=EvalInfoMask.parse(stream),
               samples_in_scan=_read_uint16(stream),
               used_channels=_read_uint16(stream),
               loop_counters=LoopCounters.parse(stream),
               cut_off=CutOffData.parse(stream),
               kspace_centre_column=_read_uint16(stream),
               coil_select=_read_uint16(stream),
               read_out_offcentre=_read_float32(stream),
               time_since_last_rf=_read_uint32(stream),
               kspace_centre_line_no=_read_uint16(stream),
               kspace_centre_partition_no=_read_uint16(stream),
               slice_data=SliceData.parse(stream),
               ice_program_para=_read_uint16(stream, 24),
               reserved_para=_read_uint16(stream, 4),
               application_counter=_read_uint16(stream),
               application_mask=_read_uint16(stream),
               crc=_read_uint32(stream))

SCAN_HEADER_SIZE = 192


@dataclasses.dataclass
class ScanData():
  """Data structure for scan data.

  Attributes:
    header: A `ScanHeader`.
    channels: A `tuple` of `ChannelData` instances.
    sync_data: A `bytes` instance containing synchronization data. Only
      set for SYNCDATA scans.
  """
  header: ScanHeader
  channels: typing.Tuple[ChannelData, ...]
  sync_data: bytes

  @classmethod
  def parse(cls, stream):
    """Parses the scan data.

    Args:
      stream: A file-like object. The stream position must be at the beginning
        of the object to be parsed.

    Returns:
      A `ScanData` instance.
    """
    header = ScanHeader.parse(stream)

    channels = []
    sync_data = None
    if header.eval_info_mask.ACQEND:
      # Nothing to read, move to end.
      stream.seek(header.dma_length - SCAN_HEADER_SIZE, os.SEEK_CUR)
    elif header.eval_info_mask.SYNCDATA:
      # Read the sync data.
      sync_data = stream.read(header.dma_length - SCAN_HEADER_SIZE)
    else:
      for _ in range(header.used_channels):
        channels.append(ChannelData.parse(stream))

    return cls(header=header, channels=channels, sync_data=sync_data)


@dataclasses.dataclass
class ParamArray():
  """A parameter holding an array value.

  Attributes:
    values: The array values.
    properties: A dictionary of properties.
  """
  values: typing.Tuple[typing.Any]
  properties: typing.Mapping[str, typing.Any]

  @classmethod
  def parse(cls, string):
    """Parses an array parameter.

    Args:
      string: The `str` to parse.

    Returns:
      A `ParamArray` instance.

    Raises:
      ValueError: If the string is not a valid bool.
    """
    string = string.strip()
    # Get properties.
    string, properties = _get_properties(string)
    # Get the raw array values.
    values = _get_array_values(string) or []

    def _set_values_in_default_map(values, default):
      output = copy.deepcopy(default)
      for k, t, v in zip(default.values.keys(),
                         default.values.values(),
                         values or itertools.repeat(None)):
        if isinstance(t, ParamMap):
          output[k] = _set_values_in_default_map(v, t)
        elif isinstance(t, ParamBool):
          output[k] = ParamBool.parse(v or '')
        elif isinstance(t, ParamChoice):
          output[k] = ParamChoice.parse(v or '')
        elif isinstance(t, ParamDouble):
          output[k] = ParamDouble.parse(v or '')
        elif isinstance(t, ParamLong):
          output[k] = ParamLong.parse(v or '')
        elif isinstance(t, ParamString):
          output[k] = ParamString.parse(v or '')
        else:
          output[k] = v
      return output

    # Set the values in the default map.
    default = properties['Default']
    if values and isinstance(default, ParamMap):
      for index, value in enumerate(values):
        values[index] = _set_values_in_default_map(value, default)

    return cls(values=values, properties=properties)

  def __getitem__(self, index):
    return self.values[index]

  def __setitem__(self, index, value):
    self.values[index] = value


@dataclasses.dataclass
class ParamBool():
  """A parameter holding a boolean value.

  Attributes:
    value: The boolean value.
    properties: A dictionary of properties.
  """
  value: bool
  properties: typing.Mapping[str, typing.Any]

  @classmethod
  def parse(cls, string):
    """Parses a bool parameter.

    Args:
      string: The `str` to parse.

    Returns:
      A `ParamBool` instance.

    Raises:
      ValueError: If the string is not a valid bool.
    """
    string = string.strip()
    # Get properties.
    string, properties = _get_properties(string)
    # If string is empty, return None.
    if not string:
      return cls(value=None, properties=properties)
    # Remove quotes if present.
    if string[0] == '"' and string[-1] == '"':
      string = string[1:-1]
    # Now parse the string.
    if string.lower() == 'true':
      value = True
    elif string.lower() == 'false':
      value = False
    else:
      raise ValueError(f"Invalid boolean value: {string}")
    return cls(value=value, properties=properties)


@dataclasses.dataclass
class ParamChoice():
  """A parameter holding a choice value.

  Attributes:
    value: The choice value.
    properties: A dictionary of properties.
  """
  value: typing.Any
  properties: typing.Mapping[str, typing.Any]

  @classmethod
  def parse(cls, string):
    """Parses a choice parameter.

    Args:
      string: The `str` to parse.

    Returns:
      A `ParamChoice` instance.
    """
    string = string.strip()
    # Get properties.
    string, properties = _get_properties(string)
    # If string is empty, return None.
    if not string:
      return cls(value=None, properties=properties)
    # Remove quotes if present.
    if string[0] == '"' and string[-1] == '"':
      string = string[1:-1]
    return cls(value=string, properties=properties)


@dataclasses.dataclass
class ParamDouble():
  """A parameter holding one or more real values.

  Attributes:
    value: The real value. May be a tuple.
    properties: A dictionary of properties.
  """
  value: typing.Union[float, typing.Tuple[float, ...]]
  properties: typing.Mapping[str, typing.Any]

  @classmethod
  def parse(cls, string):
    """Parses a real parameter.

    Args:
      string: The `str` to parse.

    Returns:
      A `ParamDouble` instance.
    """
    string = string.strip()
    # Get properties.
    string, properties = _get_properties(string)
    # If string is empty, return None.
    if not string:
      return cls(value=None, properties=properties)
    value = string.split()
    if len(value) > 1:
      value = tuple(float(s.strip()) for s in value)
    else:
      value = float(value[0])
    return cls(value=value, properties=properties)


@dataclasses.dataclass
class ParamLong():
  """A parameter holding one or more integer values.

  Attributes:
    value: The integer value. May be a tuple.
    properties: A dictionary of properties.
  """
  value: typing.Union[int, typing.Tuple[int, ...]]
  properties: typing.Mapping[str, typing.Any]

  @classmethod
  def parse(cls, string):
    """Parses an integer parameter.

    Args:
      string: The `str` to parse.

    Returns:
      A `ParamLong` instance.
    """
    string = string.strip()
    # Get properties.
    string, properties = _get_properties(string)
    # If string is empty, return None.
    if not string:
      return cls(value=None, properties=properties)
    # Support single inputs and arrays.
    value = string.split()
    if len(value) > 1:
      value = tuple(int(s.strip()) for s in value)
    else:
      value = int(value[0])
    return cls(value=value, properties=properties)


@dataclasses.dataclass
class ParamString():
  """A parameter holding a string value.

  Attributes:
    value: The string value.
    properties: A dictionary of properties.
  """
  value: str
  properties: typing.Mapping[str, typing.Any]

  @classmethod
  def parse(cls, string):
    """Parses a string parameter.

    Args:
      string: The `str` to parse.

    Returns:
      A `ParamString` instance.
    """
    string = string.strip()
    # Get properties.
    string, properties = _get_properties(string)
    # If empty, we're done.
    if not string:
      return cls(value=string, properties=properties)
    # Remove quotes if present.
    if string[0] == '"' and string[-1] == '"':
      string = string[1:-1]
    return cls(value=string, properties=properties)


@dataclasses.dataclass
class ParamMap():
  """A parameter map.

  Entries in the `values` dictionary can be directly accessed via item
  or attr getters, i.e., both `param_map['key']` and `param_map.key` are
  supported and equivalent to `param_map.values['key']`.

  Attributes:
    values: A `dict` of parameters.
    properties: A `dict` of properties.
  """
  values: typing.Mapping[str, typing.Any]
  properties: typing.Mapping[str, typing.Any]

  @classmethod
  def parse(cls, string):
    """Parses a parameter map.

    Args:
      string: The `str` to parse.

    Returns:
      A `ParamMap` instance.

    Raises:
      ValueError: If the string is invalid.
    """
    values = {}
    string, properties = _get_properties(string)

    while string.strip():
      string, param = _get_next_param(string)
      if param is None:
        raise ValueError(
            f"Could not parse next parameter in string:\n "
            f"{_head_string(string)}...")

      type_, name, value = param
      if name is not None:
        if type_ == 'ParamArray':
          values[name] = ParamArray.parse(value)
        elif type_ == 'ParamBool':
          values[name] = ParamBool.parse(value)
        elif type_ == 'ParamChoice':
          values[name] = ParamChoice.parse(value)
        elif type_ == 'ParamDouble':
          values[name] = ParamDouble.parse(value)
        elif type_ == 'ParamLong':
          values[name] = ParamLong.parse(value)
        elif type_ == 'ParamMap':
          values[name] = ParamMap.parse(value)
        elif type_ == 'ParamString':
          values[name] = ParamString.parse(value)
        else:
          warnings.warn(f'Unknown param type {type_} will not be parsed.')
          values[name] = value
    return cls(values, properties)

  def __getitem__(self, name):
    return self.values[name]

  def __setitem__(self, name, value):
    self.values[name] = value

  def __getattr__(self, name):
    if name in ('values', 'properties'):
      return super().__getattr__(name)
    if name not in self.values:
      raise AttributeError(f'Parameter `{name}` does not exist.')
    return self.values[name]


class ProtocolBuffer(ParamMap):
  """Data structure for a protocol buffer.

  This is a special type of `ParamMap`.
  """
  @classmethod
  def parse(cls, string):
    """Parses a protocol buffer.

    Args:
      string: The `str` to parse.

    Returns:
      A `ProtocolBuffer` instance.

    Raises:
      ValueError: If the string is not a valid protocol buffer.
    """
    original = string  # For error messages.

    # Get buffer properties.
    string, properties = _get_properties(string)

    # Get the top-level parameter map and begin recursive parsing.
    string, (type_, name, value) = _get_next_param(string)
    if type_ != 'ParamMap' or name != '':
      raise ValueError(f'Invalid ProtocolBuffer string: {original}')
    param_map = ParamMap.parse(value)

    return cls(param_map.values, properties)


@dataclasses.dataclass
class MeasurementHeader():
  """Data structure for a measurement header.

  Attributes:
    meas_id: The measurement ID.
    file_id: The file ID.
    offset: The start offset of the corresponding measurement, in bytes.
    length: The length of the corresponding measurement, in bytes.
    pat_name: The name of the patient.
    prot_name: The name of the protocol.
  """
  meas_id: int  # uint32
  file_id: int  # uint32
  offset: int  # uint64
  length: int  # uint64
  pat_name: str  # string (64 bytes)
  prot_name: str  # string (64 bytes)

  @classmethod
  def parse(cls, stream):
    """Parses a `MeasurementHeader` from a stream.

    Args:
      stream: A file-like object. The stream position must be at the beginning
        of the object to be parsed.

    Returns:
      A `MeasurementHeader` object.
    """
    meas_id = _read_uint32(stream)
    file_id = _read_uint32(stream)
    offset = _read_uint64(stream)
    length = _read_uint64(stream)
    pat_name = _read_string(stream, 64)
    prot_name = _read_string(stream, 64)

    return cls(meas_id=meas_id, file_id=file_id, offset=offset, length=length,
               pat_name=pat_name, prot_name=prot_name)


@dataclasses.dataclass
class MeasurementData():
  """Data structure for a measurement.

  Attributes:
    header: A `MeasurementHeader`.
    protocol: A `dict` of `ProtocolBuffer` objects.
    scans: A `tuple` of `ScanData` objects.
  """
  header: MeasurementHeader
  protocol: typing.Mapping[str, ProtocolBuffer]
  scans: typing.Tuple[ScanData, ...]

  @classmethod
  def parse(cls, stream, header):
    """Parses a `MeasurementData` from a stream.

    Args:
      stream: A file-like object. The stream position must be at the beginning
        of the object to be parsed.
      header: A `MeasurementHeader` object.

    Returns:
      A `MeasurementData` object.
    """
    # Place stream position at the beginning of the measurement.
    stream.seek(header.offset)

    # Some header info.
    prot_length = _read_uint32(stream)
    num_buffers = _read_uint32(stream)

    # Read protocol buffers.
    protocol = {}
    for _ in range(num_buffers):
      # Buffer header.
      buffer_name = _read_null_terminated_string(stream)
      buffer_length = _read_uint32(stream)

      # Read, decode and remove NUL terminator.
      buffer_string = stream.read(buffer_length).decode('latin1')[:-1]
      # Parse the protocol buffer string.
      protocol[buffer_name] = _parse_protocol_buffer(buffer_string)

    # Move to end of protocol.
    stream.seek(header.offset + prot_length)

    # Read scan data.
    scans = []
    while True:  # Continue until ACQEND.
      scans.append(ScanData.parse(stream))

      # If ACQEND is set, we're done.
      if scans[-1].header.eval_info_mask.ACQEND:
        break

    return cls(header=header, protocol=protocol, scans=scans)


MAX_RAID_FILE_MEASUREMENTS = 64
RAID_FILE_HEADER_SIZE = 152


@dataclasses.dataclass
class TwixRaidFileHeader():
  """Data structure for a TWIX raid file header.

  Attributes:
    meas_count: Number of measurements in the file.
    meas_headers: A `tuple` of `MeasurementHeader` objects.
  """
  meas_count: int  # uint32
  meas_headers: typing.Tuple[MeasurementHeader, ...]

  @classmethod
  def parse(cls, stream):
    """Parses a `TwixRaidFileHeader` from a bytes stream.

    Args:
      stream: A file-like object. The stream position must be at the beginning
        of the object to be parsed.

    Returns:
      A `TwixRaidFileHeader` instance.
    """
    _ = _read_uint32(stream)  # Ignored.
    meas_count = _read_uint32(stream)

    meas_headers = []
    for _ in range(meas_count):
      meas_headers.append(MeasurementHeader.parse(stream))

    unused_entries = MAX_RAID_FILE_MEASUREMENTS - meas_count
    stream.seek(unused_entries * RAID_FILE_HEADER_SIZE, io.SEEK_CUR)

    return cls(meas_count=meas_count, meas_headers=meas_headers)


@dataclasses.dataclass
class TwixRaidFile():
  """Data structure for a TWIX raid file.

  Attributes:
    measurements: A `tuple` of `MeasurementData` objects.
  """
  measurements: typing.Tuple[MeasurementData, ...]

  @classmethod
  def parse(cls, stream):
    """Parses a `TwixRaidFile` from a bytes stream.

    Args:
      stream: A file-like object. The stream position must be at the beginning
        of the object to be parsed.

    Returns:
      A `TwixRaidFile` instance.
    """
    header = TwixRaidFileHeader.parse(stream)

    measurements = []
    for entry in header.meas_headers:
      stream.seek(entry.offset)
      measurements.append(MeasurementData.parse(stream, entry))

    return cls(measurements=measurements)


def _parse_protocol_buffer(string):
  """Parse a protocol buffer from text.

  Args:
    string: A `str`.

  Returns:
    A `ProtocolBuffer` instance.

  Raises:
    ValueError: If the string is not a valid protocol buffer.
  """
  original = string
  # We can't parse ASCCONV for now, so remove them.
  for m in RE_ASCCONV_RANGE.finditer(string):
    string = string.replace(m.group(), '')

  # String might be empty after removing ASCCONV. If so, just return an empty
  # ProtocolBuffer.
  string = string.strip()
  if not string:
    return ProtocolBuffer(values={}, properties={})

  string, property_ = _get_next_property(string)
  if property_ is None:
    raise ValueError(
        f"Failed while parsing protocol buffer:\n "
        f"{_head_string(original)}")

  name, value = property_
  if name != 'XProtocol':
    raise ValueError(
        f"Failed while parsing protocol buffer:\n "
        f"{_head_string(original)}")

  return ProtocolBuffer.parse(value)


def _read_int16(stream, count=None):
  return _read_values(stream, 'h', count)

def _read_uint16(stream, count=None):
  return _read_values(stream, 'H', count)

def _read_int32(stream, count=None):
  return _read_values(stream, 'i', count)

def _read_uint32(stream, count=None):
  return _read_values(stream, 'I', count)

def _read_int64(stream, count=None):
  return _read_values(stream, 'q', count)

def _read_uint64(stream, count=None):
  return _read_values(stream, 'Q', count)

def _read_float32(stream, count=None):
  return _read_values(stream, 'f', count)

def _read_float64(stream, count=None):
  return _read_values(stream, 'd', count)

def _read_values(stream, fmt, count):
  fmt = f'<{count or 1}{fmt}'
  values = struct.unpack(fmt, stream.read(struct.calcsize(fmt)))
  if count is None:
    values = values[0]
  return values

def _read_tensor_data(stream, dtype, length):
  dtype = np.dtype(dtype)
  # Read data from stream.
  data = np.frombuffer(stream.getbuffer()[stream.tell():],
                       dtype=dtype, count=length // dtype.itemsize)
  # Move stream to end of data.
  stream.seek(data.nbytes, io.SEEK_CUR)
  return tf.convert_to_tensor(data)

def _read_string(stream, length):
  return stream.read(length).rstrip(NUL).decode('latin1')

def _read_null_terminated_string(stream):
  string = b''
  while True:
    char = stream.read(1)
    if char == NUL:
      break
    string += char
  return string.decode('latin1')

# The following regex patterns are used to parse the protocol buffers.
# Matches strings such as:
#   <Visible>
#   <ParamMap."">
#   <ParamBool."TestParam">
#   <ParamBool."Test;Param">
#   <ParamBool."Test@Param">
#   <ParamBool."Test Param">
RE_TAG = re.compile(r'<(?P<type>\w+)(?:\."(?P<name>[\w@; ]*)")?>')
# Matches only strings like:
#   <Visible>
RE_TAG_PROPERTY = re.compile(r'<(?P<name>\w+)>')
# Matches only strings like:
#   <ParamMap."">
#   <ParamBool."TestParam">
RE_TAG_PARAM = re.compile(r'<(?P<type>\w+)\."(?P<name>[\w@; ]*)">')
# Matches an ASCCONV range (everything between ASCCONV BEGIN AND ASCCONV END).
RE_ASCCONV_RANGE = re.compile(
    r'### ASCCONV BEGIN(?P<content>(?:.|\n)*)### ASCCONV END ###')
# Matches any whitespace character.
RE_WHITESPACE = re.compile(r'\s')
# Matches any non-whitespace character.
RE_NON_WHITESPACE = re.compile(r'\S')
# Matches a newline.
RE_NEWLINE = re.compile(r'\n')

def _get_next_element(string, re_tag):
  """Returns the next element tagged with `re_tag`.

  Args:
    string: The string to parse.
    re_tag: The regex pattern describing the tag.

  Returns:
    The string without the matched element and a tuple describing the matched
    element.

  Raises:
    ValueError: If the string is not valid.
  """
  string = string.strip()
  m = re_tag.match(string)
  if not m:
    return string, None
  try:
    type_ = m.group('type')
  except IndexError:
    # It's a property.
    type_ = None
  name = m.group('name')

  # Remove tag from string.
  string = string[m.end():]

  # Get first non-whitespace character.
  m = RE_NON_WHITESPACE.search(string)
  if not m:
    raise ValueError('Invalid string: {}'.format(string))
  c = m.group()
  string = string[m.start():]

  # Now, depending on what character we found:
  if name == "Comment":
    # Continue until the end of the line.
    m = RE_NEWLINE.search(string)
    end = m.start() if m else len(string)
    value = string[:end].strip()
  elif c == '{':
    # Find the matching closing brace.
    level = 1
    string = string[1:]
    end = None
    for end, c in enumerate(string):
      if c == '{':
        level += 1
      elif c == '}':
        level -= 1
        if level == 0:
          break
    value = string[:end].strip()
  elif c == '"':
    # Find the matching closing quote.
    string = string[1:]
    end = None
    for end, c in enumerate(string):
      if c == '"':
        break
    value = string[:end].strip()
  elif c == '<' and name == 'Default':
    # Special case for a ParamMap that defines a default value.
    string, (param_type, _, value) = _get_next_param(string)
    end = -1  # string has already been updated.
    if param_type == 'ParamMap':
      value = ParamMap.parse(value)
  else:
    # Continue until next whitespace or until the end of the string.
    m = RE_WHITESPACE.search(string)
    end = m.start() if m else len(string)
    value = string[:end].strip()

  if len(string) > 0:
    string = string[end+1:]

  if type_ is not None:
    return string, (type_, name, value)
  return string, (name, value)


def _get_next_param(string):
  """Returns the next parameter in string.

  Args:
    string: The string to parse.

  Returns:
    The string without the matched parameter and a tuple describing
    the matched parameter (its type, name and value). If no parameter
    is found, returns (string, None).
  """
  return _get_next_element(string, RE_TAG_PARAM)


def _get_next_property(string):
  """Returns the next property in string.

  Args:
    string: The string to parse.

  Returns:
    The string without the matched property and a tuple describing
    the matched property (its name and value). If no property is found,
    returns (string, None).
  """
  return _get_next_element(string, RE_TAG_PROPERTY)


def _get_properties(string):
  """Returns a list of properties found in string.

  Args:
    string: The string to parse.

  Returns:
    The string without the matched properties and a list of tuples
    describing the matched properties (their name and value).
  """
  properties = {}
  while True:
    string, property_ = _get_next_property(string)
    if property_ is None:
      break
    name, value = property_
    if name in ('Precision', 'MinSize', 'MaxSize'):
      value = int(value.strip())
    elif isinstance(value, str):
      value = value.strip()
    properties[name] = value
  return string, properties


def _get_array_values(string):
  """Retrieves raw array values.

  Args:
    string: The string to parse.

  Returns:
    A (nested) list of values, or `None` if input string is empty.
  """
  values = []
  while True:
    string = string.strip()
    if not string:
      # Job is done if string is empty. Return the values, or None if no
      # values were found.
      return values or None
    # Find the first opening brace.
    start = string.find('{')
    if start < 0:
      # No braces found. Return value as is.
      return string
    # Find the corresponding closing brace.
    level = 0
    stop = None
    for stop, c in enumerate(string):
      if c == '{':
        level += 1
      elif c == '}':
        level -= 1
        if level == 0:
          break
    value = _get_array_values(string[(start + 1):stop])
    values.append(value)
    # Update string.
    string = string[(stop + 1):]


def _head_string(string, n=10):
  """Returns the first n lines of a string.

  Args:
    string: A `str`.
    n: The number of lines to return.

  Returns:
    A `str` containing the first `n` lines of `string`.
  """
  return '\n'.join(string.splitlines()[:n])


# Some of the objects in this module are deeply nested dataclasses which can
# potentially return absurdly long string representations when printed.
# To prevent this, we customise the string representations to limit the size
# of the reprs for each field.
REPR = reprlib.Repr()

def __repr__(self):  # pylint: disable=invalid-name
  """Returns a string representation of the object.

  This repr is used for all dataclasses in this module. It is very similar to
  the default repr, except that it limits the size of the reprs for the
  field values.
  """
  fields = dataclasses.fields(self)
  return (self.__class__.__qualname__ +
          f"({', '.join(f'{f.name}={REPR.repr(getattr(self, f.name))}' for f in fields)})")  # pylint: disable=line-too-long

# Set the compressed repr for all dataclasses in this module.
for _, class_ in inspect.getmembers(sys.modules[__name__],
                                    dataclasses.is_dataclass):
  class_.__repr__ = __repr__
