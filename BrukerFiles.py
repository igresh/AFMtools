"""
Pure-Python reader for Bruker / Veeco NanoScope files.

Works on any OS (macOS, Linux, Windows) and only needs numpy. It replaces the
Windows-only ``nanoscope`` module that Bruker ships with NanoScope Analysis.

Supported files
---------------
- Image files: ``.spm`` and the numbered ``.000``, ``.001`` ... files.
  Every channel (Height Sensor, Amplitude, Phase, ...) is read and scaled to
  physical units.
- PeakForce Capture files: ``.pfc``. The height image and the force curve
  recorded at every pixel.

The scaling was checked against output from Bruker's own ``nanoscope`` module:
images match to ~1e-7 relative error, and PeakForce Capture force curves
match sample for sample.

Quick start
-----------
>>> import BrukerFiles as bf
>>> f = bf.NanoscopeFile('sample.0_00000.spm')
>>> f.channel_names
['Height Sensor', 'Amplitude', 'Phase', 'T-R Correlation']
>>> height = f.channel('Height Sensor')
>>> height.data.shape, height.z_unit, height.scan_size, height.xy_unit
((512, 512), 'nm', (20.0, 20.0), 'µm')

>>> pfc = bf.PeakForceCaptureFile('sample.000.pfc')
>>> img = pfc.image_channel.data            # height image, nm
>>> curves = pfc.force_separation_curves()  # (N_pixels, 4, 128): sep_trace, F_trace, sep_retrace, F_retrace
"""

import os
import re
import warnings

import numpy as np

__all__ = ['read_header', 'NanoscopeFile', 'PeakForceCaptureFile', 'ImageChannel',
           'load_spm', 'open_spm']


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

_HEADER_END = b'\\*File list end'

_SI_PREFIX = {'': 1.0, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6, '~': 1e-6, 'µ': 1e-6,
              'm': 1e-3, 'k': 1e3, 'M': 1e6, 'G': 1e9}

# Matches the "V" type parameter lines, e.g.
#   V [Sens. ZsensSens] (0.00000000572205 V/LSB) 10.64002 V
#   V (0.00003051758 Arb/LSB) 2.000000 Arb
_VALUE_RE = re.compile(
    r'^V\s*(?:\[(?P<soft>[^\]]*)\])?\s*'
    r'(?:\((?P<hard>[-+\d.eE]+)\s*(?P<hard_unit>.*?/LSB)\))?\s*'
    r'(?P<value>[-+\d.eE]+)\s*(?P<unit>.*)$')


def _read_raw_header(path):
    with open(path, 'rb') as fh:
        chunk = fh.read(1 << 16)
        while _HEADER_END not in chunk:
            more = fh.read(1 << 16)
            if not more:
                raise ValueError(f'{path} does not look like a Bruker NanoScope file '
                                 '(no "\\*File list end" marker).')
            chunk += more
    return chunk[:chunk.index(_HEADER_END)].decode('latin-1')


def read_header(path):
    """
    Read the text header of a NanoScope file.

    Returns
    -------
    list of (section_title, dict)
        Sections in file order (e.g. 'Ciao scan list', 'Ciao image list').
        Keys are written without the leading backslash (``'Samps/line'``,
        ``'@2:Z scale'``). If a key appears twice in a section, the first
        value is kept.
    """
    text = _read_raw_header(path).replace('\r\n', '\n').replace('\r', '\n')
    sections = []
    for block in text.split('\\*')[1:]:
        lines = block.split('\n')
        params = {}
        for line in lines[1:]:
            line = line.strip()
            if not line.startswith('\\'):
                continue
            key, sep, value = line[1:].partition(': ')
            if not sep:
                key, _, value = line[1:].partition(':')
            params.setdefault(key.strip(), value.strip())
        sections.append((lines[0].strip(), params))
    return sections


def _parse_value(text):
    """Parse a 'V [...] (... X/LSB) value unit' parameter into a dict."""
    m = _VALUE_RE.match(text.strip())
    if not m:
        return None
    out = m.groupdict()
    out['value'] = float(out['value'])
    out['hard'] = float(out['hard']) if out['hard'] is not None else None
    out['unit'] = (out['unit'] or '').strip()
    return out


def _first_number(text):
    return float(text.split()[0])


def _split_unit(unit):
    """'nm/V' -> ('nm', 'V'); 'nm' -> ('nm', '')."""
    num, _, den = unit.partition('/')
    return num.strip(), den.strip()


def _prefix_factor(unit, base):
    """Factor to convert ``unit`` (e.g. 'mV') into ``base`` (e.g. 'V')."""
    if unit == base:
        return 1.0
    if unit.endswith(base):
        prefix = unit[:-len(base)]
        if prefix in _SI_PREFIX:
            return _SI_PREFIX[prefix]
    return 1.0


def _pretty_unit(unit):
    return unit.replace('~m', 'µm').replace('º', '°')


# ---------------------------------------------------------------------------
# Generic file
# ---------------------------------------------------------------------------

class ImageChannel:
    """One image channel of a NanoScope file. ``data`` is in physical units."""

    def __init__(self, nfile, params, index):
        self.file = nfile
        self.params = params
        self.index = index

        self.data_offset = int(params['Data offset'])
        self.data_length = int(params['Data length'])
        self.bytes_per_pixel = int(params.get('Bytes/pixel', 2))
        self.samples_per_line = int(params['Samps/line'].split()[0])
        self.number_of_lines = int(params['Number of lines'])
        self.line_direction = params.get('Line Direction', '')
        self.frame_direction = params.get('Frame direction', '')

        key = _find_key(params, 'Image Data')
        m = re.search(r'\[(.*?)\]\s*"(.*?)"', params.get(key, '')) if key else None
        self.data_type = m.group(1) if m else ''
        self.data_type_desc = m.group(2) if m else params.get('Data Type Description', '')

        ss = params.get('Scan Size', '').split()
        if len(ss) >= 3:
            self.scan_size = (float(ss[0]), float(ss[1]))
            self.xy_unit = _pretty_unit(ss[2])
        else:
            self.scan_size, self.xy_unit = (np.nan, np.nan), ''

        ar = params.get('Aspect Ratio', '1:1').split(':')
        try:
            self.aspect_ratio = float(ar[0]) / float(ar[1])
        except (ValueError, IndexError, ZeroDivisionError):
            self.aspect_ratio = 1.0

        self._compute_scale()

    # -- scaling ---------------------------------------------------------
    def _compute_scale(self):
        key = _find_key(self.params, 'Z scale')
        zs = _parse_value(self.params[key]) if key else None
        if zs is None:
            self.scale, self.z_unit, self.z_scale_in_sw_units = 1.0, 'LSB', np.nan
            return

        # NanoScope convention: the raw range spans the 'Z scale' value.
        # Bruker's own software divides by 2**(8*bytes) - 2 (verified against
        # the nanoscope module). Then multiply by the soft scale (sensitivity).
        per_lsb = zs['value'] / (2.0 ** (8 * self.bytes_per_pixel) - 2)
        unit = zs['unit']
        if zs['soft']:
            sens = self.file.sensitivity(zs['soft'])
            if sens is not None:
                sval, sunit = sens
                num, den = _split_unit(sunit)
                per_lsb *= sval * _prefix_factor(unit, den or unit)
                unit = num if num else unit
                self.z_scale_in_sw_units = zs['value'] * sval * _prefix_factor(zs['unit'], den or zs['unit'])
            else:
                self.z_scale_in_sw_units = zs['value']
        else:
            self.z_scale_in_sw_units = zs['value']
        self.scale = per_lsb
        self.z_unit = _pretty_unit(unit)

    # -- data ------------------------------------------------------------
    @property
    def raw(self):
        """Raw integer data, shape (lines, samples_per_line)."""
        n = self.samples_per_line * self.number_of_lines
        # Some channels declare 2 bytes/pixel but are stored as 4 bytes;
        # the data length is authoritative.
        bpp = self.data_length // n if self.data_length >= n else self.bytes_per_pixel
        dtype = {1: '<i1', 2: '<i2', 4: '<i4'}[bpp]
        arr = np.fromfile(self.file.path, dtype=dtype, count=n, offset=self.data_offset)
        return arr.reshape(self.number_of_lines, self.samples_per_line)

    @property
    def data(self):
        """Scaled image (float64), shape (lines, samples_per_line)."""
        return self.raw.astype(np.float64) * self.scale

    # nanoscope-module style aliases
    def create_image(self):
        return self.data

    @property
    def metadata(self):
        return {'ScanSize': self.scan_size[0],
                'ScanPoints': self.number_of_lines,
                'Type': self.data_type_desc,
                'direction': self.line_direction,
                'aspect ratio': self.aspect_ratio,
                'z unit': self.z_unit,
                'xy unit': self.xy_unit}

    def __repr__(self):
        return (f'<ImageChannel {self.index}: {self.data_type_desc!r} '
                f'{self.number_of_lines}x{self.samples_per_line}, '
                f'{self.scan_size[0]:g} {self.xy_unit}, z in {self.z_unit}>')


def _find_key(params, suffix):
    """Find keys like '@2:Z scale' or '@4:Image Data'."""
    for k in params:
        if k.lstrip('@').split(':')[-1] == suffix:
            return k
    return None


class NanoscopeFile:
    """
    A Bruker NanoScope file (.spm, .000, .pfc, ...).

    Attributes
    ----------
    path : str
    header : list of (title, dict)
    channels : list of ImageChannel
    """

    def __init__(self, path):
        self.path = os.fspath(path)
        self.header = read_header(self.path)
        self._sens = {}
        for _, params in self.header:
            for k, v in params.items():
                if k.startswith('@Sens.'):
                    pv = _parse_value(v)
                    if pv is not None:
                        self._sens.setdefault(k[len('@Sens.'):].strip(), (pv['value'], pv['unit']))
        self.channels = [ImageChannel(self, p, i)
                         for i, p in enumerate(self.sections('Ciao image list'))]

    # context-manager support, so ``with NanoscopeFile(f) as file:`` works
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __len__(self):
        return len(self.channels)

    def __iter__(self):
        return iter(self.channels)

    def sections(self, title):
        """All header sections with the given title."""
        return [p for t, p in self.header if t == title]

    def section(self, title):
        s = self.sections(title)
        return s[0] if s else {}

    def param(self, key, default=None):
        """First value of ``key`` found anywhere in the header."""
        for _, p in self.header:
            if key in p:
                return p[key]
        return default

    def sensitivity(self, name):
        """
        Look up a sensitivity ('soft scale'). ``name`` may be given with or
        without the 'Sens. ' prefix, e.g. 'Sens. ZsensSens' or 'DeflSens'.
        Returns (value, unit) or None.
        """
        name = name.strip()
        if name.startswith('Sens.'):
            name = name[len('Sens.'):].strip()
        return self._sens.get(name)

    @property
    def microscope_mode(self):
        """Acquisition mode, e.g. 'Tapping', 'Contact', 'PeakForce QNM'."""
        m = re.search(r'"(.*)"', self.param('@MicroscopeList', ''))
        return m.group(1) if m else ''

    @property
    def has_force_data(self):
        return any(p.get('Data type') == 'FORCE' for p in self.sections('Ciao force image list'))

    @property
    def is_peakforce(self):
        """True if the file was acquired in a PeakForce-based mode (sinusoidal Z)."""
        mode = self.param('@MicroscopeList', '')
        return bool(re.search(r'peak\s*force|softharmonix|scanasyst', mode, re.I))

    @property
    def channel_names(self):
        return [c.data_type_desc for c in self.channels]

    def channel(self, name_or_index):
        """Get a channel by index or by name ('Height Sensor', 'Phase', ...)."""
        if isinstance(name_or_index, int):
            return self.channels[name_or_index]
        for c in self.channels:
            if name_or_index in (c.data_type_desc, c.data_type):
                return c
        raise KeyError(f'No channel {name_or_index!r}. Available: {self.channel_names}')

    def __repr__(self):
        return f'<NanoscopeFile {os.path.basename(self.path)}: {self.channel_names}>'


# ---------------------------------------------------------------------------
# PeakForce Capture
# ---------------------------------------------------------------------------

class PeakForceCaptureFile(NanoscopeFile):
    """
    A PeakForce Capture (.pfc) file: one height image plus the full
    PeakForce tapping cycle recorded at every pixel.

    Each raw curve holds one full sinusoidal Z period. Following NanoScope
    Analysis, the curve is rotated by the 'Sync Distance' so the approach
    (trace) ends and the retract (retrace) starts at maximum extension. The
    Z position is a sinusoid with amplitude equal to the 'Peak Force
    Amplitude'. Separation = Z + deflection, zeroed at the end of the trace.
    """

    INVALID = -32768

    def __init__(self, path):
        super().__init__(path)
        fsecs = self.sections('Ciao force image list')
        if not fsecs:
            raise ValueError(f'{self.path} has no force curve data (not a PeakForce Capture file).')
        if not self.is_peakforce:
            raise ValueError(
                f'{self.path} contains force curves, but was acquired in '
                f'{self.microscope_mode or "an unknown"} mode, not PeakForce. '
                'PeakForceCaptureFile assumes the sinusoidal PeakForce Z motion and would '
                'give wrong separations for this file. Its images can still be read '
                'with NanoscopeFile.')
        self.force_params = fsecs[0]
        fp = self.force_params

        self.image_channel = self.channels[0] if self.channels else None
        self.spring_constant = float(fp['Spring Constant'])
        self._force_offset = int(fp['Data offset'])
        self._force_length = int(fp['Data length'])
        self._force_bpp = int(fp.get('Bytes/pixel', 2))

        spl = [int(s) for s in fp['Samps/line'].split()]
        self.samples_trace = spl[0]
        self.samples_retrace = spl[1] if len(spl) > 1 else spl[0]
        if self.samples_trace != self.samples_retrace:
            raise ValueError(
                f'{self.path}: trace and retrace have different lengths '
                f'({self.samples_trace} vs {self.samples_retrace}); this is not the '
                'PeakForce Capture layout this reader was validated on.')
        self.samples_per_curve = self.samples_trace + self.samples_retrace
        self.number_of_force_curves = self._force_length // (self._force_bpp * self.samples_per_curve)

        # Deflection scaling: raw LSB -> V (hard scale) -> nm (DeflSens)
        zs = _parse_value(fp[_find_key(fp, 'Z scale')])
        sval, sunit = self.sensitivity(zs['soft'] or 'DeflSens')
        num, den = _split_unit(sunit)
        hard_unit = zs['hard_unit'].split('/')[0] if zs['hard_unit'] else den
        self.deflection_sensitivity = sval                        # nm/V
        self.deflection_per_lsb = zs['hard'] * sval * _prefix_factor(hard_unit, den)  # nm/LSB
        self.force_per_lsb = self.deflection_per_lsb * self.spring_constant           # nN/LSB

        self.sync_distance = float(self.param('Sync Distance'))
        self.peakforce_amplitude = float(self.param('Peak Force Amplitude'))   # nm
        self.peakforce_frequency = self.param('PFT Freq')
        # 'Sync Distance' is counted in ticks of a fixed 512 kHz clock
        # (= curve samples at 2 kHz with 256 samples/curve). Convert to curve samples.
        # Verified against nanoscope output at 2 kHz; at 1 kHz the converted value
        # lands on the measured deflection peak.
        self.sync_samples = self.sync_distance * self._pft_hz() * self.samples_per_curve / 512e3

        # Values the old PeakforceImport saved to values.csv
        ic = self.image_channel
        self.scan_size = ic.scan_size[0] if ic else np.nan
        self.scan_size_unit = ic.xy_unit if ic else ''
        self.number_of_lines = ic.number_of_lines if ic else 0
        self.samples_per_line = ic.samples_per_line if ic else 0

    # -- raw curves ------------------------------------------------------
    def raw_curves(self, pixels=None):
        """Raw int16 curves as stored, shape (n_pixels, samples_per_curve)."""
        n, s = self.number_of_force_curves, self.samples_per_curve
        dtype = {2: '<i2', 4: '<i4'}[self._force_bpp]
        mm = np.memmap(self.path, dtype=dtype, mode='r', offset=self._force_offset, shape=(n, s))
        return np.array(mm if pixels is None else mm[pixels])

    def _time_ordered(self, raw):
        """Stored as [first half reversed, second half] -> one continuous period."""
        h = self.samples_trace
        t = np.concatenate([raw[..., :h][..., ::-1], raw[..., h:]], axis=-1).astype(np.float64)
        # Replace invalid samples with the mean of their neighbours (as NanoScope does)
        bad = t == self.INVALID
        if bad.any():
            prev_ = np.roll(t, 1, axis=-1)
            next_ = np.roll(t, -1, axis=-1)
            t[bad] = 0.5 * (prev_[bad] + next_[bad])
        return t

    def _pft_hz(self):
        m = re.match(r'\s*([\d.]+)\s*(k?)hz', str(self.peakforce_frequency), re.I)
        if not m:
            return 2e3
        return float(m.group(1)) * (1e3 if m.group(2) else 1.0)

    @property
    def _start(self):
        """Index in the time-ordered period where the retrace starts."""
        return int(round(self.sync_samples)) + 1

    def deflection_curves(self, pixels=None):
        """
        Deflection (nm) for each pixel.

        Returns (trace, retrace), each of shape (n_pixels, samples). The trace
        is the approach (ends at max extension), the retrace is the retract.
        """
        t = self._time_ordered(self.raw_curves(pixels)) * self.deflection_per_lsb
        n = self.samples_per_curve
        idx_retrace = (self._start + np.arange(self.samples_retrace)) % n
        idx_trace = (self._start + self.samples_retrace + np.arange(self.samples_trace)) % n
        return t[..., idx_trace], t[..., idx_retrace]

    def z_ramp(self):
        """Z position (nm) for trace and retrace; larger = further from the surface."""
        n = self.samples_per_curve
        A = self.peakforce_amplitude
        idx_retrace = self._start + np.arange(self.samples_retrace)
        idx_trace = self._start + self.samples_retrace + np.arange(self.samples_trace)
        z = lambda i: -A * np.cos(2 * np.pi * (i - self.sync_samples) / n)
        return z(idx_trace), z(idx_retrace)

    def force_z_curves(self, pixels=None):
        """
        Force vs Z for each pixel (nN, nm).

        Returns array of shape (n_pixels, 4, samples):
        [z_trace, force_trace, z_retrace, force_retrace]. Z is shifted so
        the minimum is 0.
        """
        dt, dr = self.deflection_curves(pixels)
        zt, zr = self.z_ramp()
        zmin = min(zt.min(), zr.min())
        zt, zr = zt - zmin, zr - zmin
        k = self.spring_constant
        out = np.empty((dt.shape[0], 4, dt.shape[-1]))
        out[:, 0], out[:, 1], out[:, 2], out[:, 3] = zt, dt * k, zr, dr * k
        return out

    def force_separation_curves(self, pixels=None):
        """
        Force vs tip-sample separation for each pixel (nN, nm).

        Returns array of shape (n_pixels, 4, samples):
        [sep_trace, force_trace, sep_retrace, force_retrace]. This matches
        what the old Windows ``PeakforceImport`` saved to qnmcurves.npy.
        Separation = Z + deflection, zeroed at the last point of the trace.
        """
        dt, dr = self.deflection_curves(pixels)
        zt, zr = self.z_ramp()
        st, sr = zt + dt, zr + dr
        offset = st[:, -1:].copy()
        k = self.spring_constant
        out = np.empty((dt.shape[0], 4, dt.shape[-1]))
        out[:, 0], out[:, 1], out[:, 2], out[:, 3] = st - offset, dt * k, sr - offset, dr * k
        return out

    def pixel_index(self, row, col):
        """Pixel index of image position (row, col)."""
        return row * self.samples_per_line + col

    def __repr__(self):
        return (f'<PeakForceCaptureFile {os.path.basename(self.path)}: '
                f'{self.number_of_force_curves} curves x {self.samples_per_curve} samples, '
                f'k={self.spring_constant} N/m>')


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def open_spm(path):
    """
    Open any NanoScope file.

    Returns PeakForceCaptureFile for PeakForce Capture files, otherwise
    NanoscopeFile. Files holding force curves from a non-PeakForce mode
    (force volume, force ramps) open as NanoscopeFile with a warning:
    their images are fine, but their force curves are not supported.
    """
    nf = NanoscopeFile(path)
    if nf.has_force_data:
        if nf.is_peakforce:
            return PeakForceCaptureFile(path)
        warnings.warn(
            f'{os.path.basename(nf.path)} contains force curves from '
            f'{nf.microscope_mode or "an unknown"} mode. Only PeakForce Capture curves '
            'are supported; opening the image channels only.', stacklevel=2)
    return nf


def load_spm(path):
    """
    Load every image channel of a NanoScope image file.

    Returns a dict in the same layout the old nanoscope-based DataImporter
    notebooks produced:
        {channel_index: {'image': ndarray, 'metadata': dict}}
    """
    nf = NanoscopeFile(path)
    return {c.index: {'image': c.data, 'metadata': c.metadata} for c in nf.channels}
