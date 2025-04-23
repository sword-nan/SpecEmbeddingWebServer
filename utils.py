from pathlib import Path

import numpy as np
import numpy.typing as npt
from numba import prange, njit
from matchms.importing import load_from_mgf, load_from_msp, load_from_mzxml
from matchms.filtering import default_filters, normalize_intensities

def read_raw_spectra(path: str):
    suffix = Path(path).suffix
    if suffix == ".mgf":
        spectra = list(load_from_mgf(path))
    elif suffix == ".msp":
        spectra = list(load_from_msp(path))
    elif suffix == ".mzxml":
        spectra = list(load_from_mzxml(path))
    else:
        raise ValueError(f"Not support the {suffix} format")
    
    spectra = [default_filters(s) for s in spectra]
    spectra = [normalize_intensities(s) for s in spectra]
    return spectra

@njit
def cosine_similarity(A: npt.NDArray, B: npt.NDArray):
    norm_A = np.sqrt(np.sum(A ** 2, axis=1)) + 1e-8
    norm_B = np.sqrt(np.sum(B ** 2, axis=1)) + 1e-8
    normalize_A = A / norm_A[:, np.newaxis]
    normalize_B = B / norm_B[:, np.newaxis]
    scores = np.dot(normalize_A, normalize_B.T)
    return scores

@njit(parallel=True)
def top_k_indices(score, top_k):
    rows, cols = score.shape
    indices = np.empty((rows, top_k), dtype=np.int64)
    for i in prange(rows):
        row = score[i]
        sorted_idx = np.argsort(row)[::-1]
        indices[i] = sorted_idx[:top_k]
    return indices