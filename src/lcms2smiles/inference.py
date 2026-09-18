from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .chemistry import formula_vector
from .data import ADDUCTS, INSTRUMENTS, preprocess_peaks
from .decoding import generate_and_rank
from .models.model import SpectrumToSmiles
from .tokenizer import SmilesTokenizer
from .utils import load_config


def build_batch(record: dict, cfg: dict, device: torch.device) -> dict:
    d = cfg["data"]
    mz, intensity, peak_mask = preprocess_peaks(
        np.asarray(record["mzs"], dtype=np.float32),
        np.asarray(record["intensities"], dtype=np.float32),
        d["max_peaks"], d["min_relative_intensity"], d["intensity_power"],
    )
    formula = record.get("formula") or ""
    return {
        "mz": torch.from_numpy(mz).unsqueeze(0).to(device),
        "intensity": torch.from_numpy(intensity).unsqueeze(0).to(device),
        "peak_mask": torch.from_numpy(peak_mask).unsqueeze(0).to(device),
        "precursor_mz": torch.tensor([float(record["precursor_mz"])], device=device),
        "collision_energy": torch.tensor([float(record.get("collision_energy") or 0.0)], device=device),
        "adduct_id": torch.tensor([ADDUCTS.get(record.get("adduct", "<unk>"), 0)], dtype=torch.long, device=device),
        "instrument_id": torch.tensor([INSTRUMENTS.get(record.get("instrument_type", "<unk>"), 0)], dtype=torch.long, device=device),
        "formula_vec": torch.from_numpy(formula_vector(formula)).unsqueeze(0).to(device),
        "adduct": [record.get("adduct", "<unk>")],
        "formula": [formula],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict ranked SMILES from a JSON LC-MS/MS spectrum")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="JSON file containing mzs, intensities, precursor_mz and metadata")
    ap.add_argument("--output", default="-")
    args = ap.parse_args()
    cfg = load_config(args.config)
    record = json.loads(Path(args.input).read_text(encoding="utf-8"))
    tok = SmilesTokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectrumToSmiles(vocab_size=len(tok.vocab), pad_id=tok.pad_id, **cfg["model"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    batch = build_batch(record, cfg, device)
    dec = cfg["decoding"]; w = dec["score_weights"]
    cands = generate_and_rank(model, batch, tok, dec["beam_size"], dec["max_length"], dec["top_k"], dec["mass_tolerance_ppm"], dec["require_formula_match_when_available"], w["sequence"], w["fingerprint"], w["mass"])
    payload = {"input": record, "candidates": [c.__dict__ for c in cands]}
    text = json.dumps(payload, indent=2)
    if args.output == "-":
        print(text)
    else:
        out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
