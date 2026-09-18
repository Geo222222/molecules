from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import MassSpecGymDataset
from .models.model import SpectrumToSmiles
from .tokenizer import SmilesTokenizer
from .utils import load_config, move_batch, seed_everything


def build_model(cfg: dict, tok: SmilesTokenizer) -> SpectrumToSmiles:
    m = cfg["model"]
    return SpectrumToSmiles(vocab_size=len(tok.vocab), pad_id=tok.pad_id, **m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    tok = SmilesTokenizer()
    d = cfg["data"]
    m = cfg["model"]
    train_ds = MassSpecGymDataset(
        d["path"], "train", tok, d["max_peaks"], d["max_smiles_tokens"],
        d["min_relative_intensity"], d["intensity_power"], m["fingerprint_bits"],
    )
    val_ds = MassSpecGymDataset(
        d["path"], "val", tok, d["max_peaks"], d["max_smiles_tokens"],
        d["min_relative_intensity"], d["intensity_power"], m["fingerprint_bits"],
    )
    t = cfg["training"]
    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, num_workers=t["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=t["batch_size"], shuffle=False, num_workers=t["num_workers"], pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, tok).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    use_amp = bool(t["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ckpt_dir = Path(t["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")

    def run(loader, training: bool) -> float:
        model.train(training)
        total, count = 0.0, 0
        for batch in loader:
            batch = move_batch(batch, device)
            # Formula dropout prevents the network from treating the exact formula as an oracle.
            # It trains both the formula-assisted and realistic formula-free inference regimes.
            if training and t.get("formula_dropout", 0.0) > 0:
                drop = torch.rand(batch["formula_vec"].size(0), device=device) < t["formula_dropout"]
                batch["formula_vec"] = batch["formula_vec"].clone()
                batch["formula_vec"][drop] = 0
            target = batch["smiles_ids"][:, 1:]
            if training:
                opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits, fp_logits = model(batch)
                seq_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), target.reshape(-1),
                    ignore_index=tok.pad_id, label_smoothing=t["label_smoothing"],
                )
                fp_loss = F.binary_cross_entropy_with_logits(fp_logits, batch["fingerprint"])
                loss = seq_loss + t["fingerprint_loss_weight"] * fp_loss
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
                scaler.step(opt)
                scaler.update()
            total += float(loss.detach()) * target.size(0)
            count += target.size(0)
        return total / max(count, 1)

    for epoch in range(1, t["epochs"] + 1):
        tr = run(train_loader, True)
        with torch.no_grad():
            va = run(val_loader, False)
        print(f"epoch={epoch:03d} train_loss={tr:.5f} val_loss={va:.5f}")
        payload = {"model": model.state_dict(), "config": cfg, "epoch": epoch, "val_loss": va}
        torch.save(payload, ckpt_dir / "last.pt")
        if va < best:
            best = va
            torch.save(payload, ckpt_dir / "best.pt")


if __name__ == "__main__":
    main()
