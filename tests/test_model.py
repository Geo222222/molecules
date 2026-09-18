import torch
from lcms2smiles.models.model import SpectrumToSmiles
from lcms2smiles.tokenizer import SmilesTokenizer


def test_forward_shapes():
    tok = SmilesTokenizer()
    model = SpectrumToSmiles(len(tok.vocab), tok.pad_id, d_model=64, nhead=4, encoder_layers=1, decoder_layers=1, dim_feedforward=128, max_peaks=8, max_smiles_tokens=16, fingerprint_bits=32)
    batch = {
        "mz": torch.rand(2, 8) * 500,
        "intensity": torch.rand(2, 8),
        "peak_mask": torch.zeros(2, 8, dtype=torch.bool),
        "precursor_mz": torch.tensor([300.0, 400.0]),
        "collision_energy": torch.tensor([20.0, 40.0]),
        "adduct_id": torch.ones(2, dtype=torch.long),
        "instrument_id": torch.ones(2, dtype=torch.long),
        "formula_vec": torch.zeros(2, 13),
        "smiles_ids": torch.tensor([tok.encode("CCO", 16), tok.encode("CCN", 16)]),
    }
    logits, fp = model(batch)
    assert logits.shape == (2, 15, len(tok.vocab))
    assert fp.shape == (2, 32)
