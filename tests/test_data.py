import numpy as np
from lcms2smiles.data import preprocess_peaks


def test_preprocess_peaks_shapes_and_order():
    mz = np.array([200.0, 50.0, 100.0])
    ints = np.array([0.1, 1.0, 0.5])
    m, i, mask = preprocess_peaks(mz, ints, max_peaks=5)
    assert m.shape == i.shape == mask.shape == (5,)
    assert list(m[:3]) == [50.0, 100.0, 200.0]
    assert mask.tolist() == [False, False, False, True, True]
