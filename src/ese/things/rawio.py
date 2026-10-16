"""THINGS-EEG2 raw file reader with a restricted unpickler and the test-file seal (D-111 decision 4 and 9, F-V).

The raw files (raw_eeg_training.npy, raw_eeg_test.npy, image_metadata.npy) are .npy files holding a pickled 0-d object array
with a dict inside. np.load(allow_pickle=True) would run arbitrary pickle code, and `python -I` does not prevent that, so the
files are read here instead: the .npy header is parsed with numpy.lib.format (dtype object, shape ()), and the pickle stream is
read by an Unpickler whose find_class allows only numpy array reconstruction (_reconstruct, ndarray, dtype, scalar). Any
other global stops the read. The dict keys are compared with the expected set of the file kind; other keys stop the read.

Seal: a path whose basename is the sealed test name (raw_eeg_test.npy) is read only with unseal=True (the verdict job's
explicit flag) and only if its sha256 equals the row of the seal table (seal_test_sha256.tsv, eeg-dir3). Every other caller
(training, development, smoke, preprocessing) stops with an AssertionError before the file is opened.
"""
import os, io, csv, pickle, hashlib
import numpy as np
from numpy.lib import format as npformat

TEST_NAME = 'raw_eeg_test.npy'
KEYS = {'raw': {'ch_names', 'sfreq', 'ch_types', 'raw_eeg_data'}}
META_REQUIRED = {'train_img_concepts', 'train_img_files', 'test_img_concepts', 'test_img_files'}
META_OPTIONAL = {'train_img_concepts_THINGS', 'test_img_concepts_THINGS'}

_ALLOWED = {}
for _mod in ('numpy.core.multiarray', 'numpy._core.multiarray'):
    _ALLOWED[(_mod, '_reconstruct')] = np._core.multiarray._reconstruct
    _ALLOWED[(_mod, 'scalar')] = np._core.multiarray.scalar
_ALLOWED[('numpy', 'ndarray')] = np.ndarray
_ALLOWED[('numpy', 'dtype')] = np.dtype


class SealError(AssertionError):
    pass


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _ALLOWED:
            raise pickle.UnpicklingError(f'global {module}.{name} is not allowed in a THINGS-EEG2 file')
        return _ALLOWED[(module, name)]


def is_test(path): return os.path.basename(path) == TEST_NAME


def guard(path, unseal=False, seal_table=None):
    """stop unless `path` may be opened: non-test files always; the test file only with unseal and a matching sha256"""
    if not is_test(path): return
    if not unseal: raise SealError(f'{path}: sealed test file (D-111), opened outside the verdict job')
    assert seal_table, 'unseal needs the seal table path'
    rel = os.path.join(*os.path.normpath(path).split(os.sep)[-3:])          # sub-XX/ses-YY/raw_eeg_test.npy
    want = read_seal(seal_table).get(rel)
    if want is None: raise SealError(f'{rel}: not in the seal table {seal_table}')
    got = sha256(path)
    if got != want: raise SealError(f'{rel}: sha256 {got} differs from the sealed {want}')


def read_seal(table):
    """seal table rows -> {sub-XX/ses-YY/raw_eeg_test.npy: sha256}; a row is any tab-separated line holding a test-file path
    and a 64-hex sha256 (header lines and other columns are ignored)"""
    out = {}
    for row in csv.reader(open(table), delimiter='\t'):
        p = next((x for x in row if x.endswith(TEST_NAME)), None); h = next((x.lower() for x in row if len(x) == 64 and all(c in '0123456789abcdefABCDEF' for c in x)), None)
        if p and h: out[os.path.join(*os.path.normpath(p).split(os.sep)[-3:])] = h
    return out


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''): h.update(b)
    return h.hexdigest()


def load_dict(path, kind='raw', unseal=False, seal_table=None):
    """read one pickled-dict .npy file with the restricted unpickler; returns the dict (keys checked against KEYS[kind])"""
    guard(path, unseal, seal_table)
    with open(path, 'rb') as f:
        version = npformat.read_magic(f)
        shape, fortran, dtype = (npformat.read_array_header_1_0 if version == (1, 0) else npformat.read_array_header_2_0)(f, max_header_size=10000)
        assert dtype == np.dtype(object) and shape == (), f'{path}: expected a pickled 0-d object array, got {dtype} {shape}'
        arr = RestrictedUnpickler(f).load()
    assert isinstance(arr, np.ndarray) and arr.shape == () and arr.dtype == object, f'{path}: unexpected outer object {type(arr)}'
    d = arr.item(); assert isinstance(d, dict), f'{path}: content is {type(d)}, not a dict'
    if kind in KEYS:
        assert set(d) == KEYS[kind], f'{path}: keys {sorted(d)} differ from the expected {sorted(KEYS[kind])}'
        check_raw(d, path)
    elif kind == 'meta':                                                     # image_metadata.npy
        assert META_REQUIRED <= set(d) <= META_REQUIRED | META_OPTIONAL, f'{path}: keys {sorted(d)} outside {sorted(META_REQUIRED | META_OPTIONAL)}'
        for k, v in d.items(): assert all(isinstance(s, str) for s in list(v)), f'{path}: {k} holds non-string entries'
    else:
        raise ValueError(f'unknown file kind {kind}')
    return d


def check_raw(d, path):
    x = d['raw_eeg_data']; assert isinstance(x, np.ndarray) and x.ndim == 2 and x.dtype.kind == 'f', f'{path}: raw_eeg_data {type(x)} {getattr(x, "dtype", None)}'
    C = x.shape[0]; assert len(d['ch_names']) == C and len(d['ch_types']) == C, f'{path}: {C} rows, {len(d["ch_names"])} names, {len(d["ch_types"])} types'
    assert all(isinstance(n, str) for n in d['ch_names']) and all(isinstance(t, str) for t in d['ch_types']), f'{path}: non-string channel names or types'
    assert float(d['sfreq']) > 0, f'{path}: sfreq {d["sfreq"]}'
