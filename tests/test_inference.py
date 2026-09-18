import torch
from lcms2smiles.inference import build_batch


def test_build_batch_formula_free():
    cfg = {"data": {"max_peaks": 8, "min_relative_intensity": 0.001, "intensity_power": 0.5}}
    rec = {"mzs": [50.0, 100.0], "intensities": [1.0, 0.5], "precursor_mz": 151.0, "adduct": "[M+H]+"}
    batch = build_batch(rec, cfg, torch.device("cpu"))
    assert batch["mz"].shape == (1, 8)
    assert batch["formula_vec"].sum().item() == 0.0
