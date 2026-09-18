from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from .chemistry import canonical_smiles
from .data import MassSpecGymDataset
from .decoding import generate_and_rank
from .metrics import evaluate_candidates, scaffold
from .models.model import SpectrumToSmiles
from .tokenizer import SmilesTokenizer
from .utils import load_config, move_batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--fold", default="test", choices=["val", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", default="artifacts/evaluation.json")
    ap.add_argument("--formula-free", action="store_true", help="Evaluate without ground-truth molecular formula")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tok = SmilesTokenizer()
    d, m = cfg["data"], cfg["model"]
    ds = MassSpecGymDataset(d["path"], args.fold, tok, d["max_peaks"], d["max_smiles_tokens"], d["min_relative_intensity"], d["intensity_power"], m["fingerprint_bits"])
    if args.limit:
        ds = Subset(ds, range(min(args.limit, len(ds))))
    raw = pd.read_csv(d["path"], sep="\t")
    train = raw[raw.fold == "train"].smiles.astype(str).tolist()
    train_smiles = {canonical_smiles(s) or s for s in train}
    train_scaffolds = {scaffold(s) for s in train_smiles}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectrumToSmiles(vocab_size=len(tok.vocab), pad_id=tok.pad_id, **m).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    dec, rows = cfg["decoding"], []
    w = dec["score_weights"]
    for batch in DataLoader(ds, batch_size=1, shuffle=False):
        target = batch["smiles"][0]
        moved = move_batch(batch, device)
        if args.formula_free:
            moved["formula_vec"] = torch.zeros_like(moved["formula_vec"])
            moved["formula"] = [""]
        cands = generate_and_rank(model, moved, tok, dec["beam_size"], dec["max_length"], dec["top_k"], dec["mass_tolerance_ppm"], dec["require_formula_match_when_available"], w["sequence"], w["fingerprint"], w["mass"])
        rec = evaluate_candidates(target, [c.canonical_smiles for c in cands], train_smiles, train_scaffolds).to_dict()
        rec["identifier"] = batch["identifier"][0]
        rows.append(rec)
    df = pd.DataFrame(rows)
    summary = {k: float(df[k].mean()) for k in df.columns if k != "identifier"} if len(df) else {}
    payload = {"fold": args.fold, "mode": "formula_free" if args.formula_free else "formula_assisted", "n": len(rows), "summary": summary, "records": rows}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
