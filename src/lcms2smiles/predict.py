from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from .data import MassSpecGymDataset
from .decoding import generate_and_rank
from .models.model import SpectrumToSmiles
from .tokenizer import SmilesTokenizer
from .utils import load_config, move_batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--fold", default="test", choices=["train", "val", "test"])
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--formula-free", action="store_true", help="Hide formula from the model and reranker")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tok = SmilesTokenizer()
    d, m = cfg["data"], cfg["model"]
    ds = MassSpecGymDataset(d["path"], args.fold, tok, d["max_peaks"], d["max_smiles_tokens"], d["min_relative_intensity"], d["intensity_power"], m["fingerprint_bits"])
    batch = next(iter(DataLoader(torch.utils.data.Subset(ds, [args.index]), batch_size=1)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectrumToSmiles(vocab_size=len(tok.vocab), pad_id=tok.pad_id, **m).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    moved = move_batch(batch, device)
    if args.formula_free:
        moved["formula_vec"] = torch.zeros_like(moved["formula_vec"])
        moved["formula"] = [""]
    dec = cfg["decoding"]
    w = dec["score_weights"]
    cands = generate_and_rank(model, moved, tok, dec["beam_size"], dec["max_length"], dec["top_k"], dec["mass_tolerance_ppm"], dec["require_formula_match_when_available"], w["sequence"], w["fingerprint"], w["mass"])
    print(json.dumps({"target": batch["smiles"][0], "identifier": batch["identifier"][0], "candidates": [c.__dict__ for c in cands]}, indent=2))


if __name__ == "__main__":
    main()
