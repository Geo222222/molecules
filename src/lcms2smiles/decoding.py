from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .chemistry import canonical_smiles, exact_mass, molecular_formula, neutral_mass_from_precursor, ppm_error, morgan_array
from .tokenizer import SmilesTokenizer


@dataclass
class Candidate:
    smiles: str
    canonical_smiles: str
    sequence_score: float
    fingerprint_score: float
    mass_error_ppm: float | None
    formula_match: bool | None
    total_score: float


@torch.no_grad()
def beam_generate(
    model,
    batch: dict[str, torch.Tensor],
    tokenizer: SmilesTokenizer,
    beam_size: int = 20,
    max_length: int = 192,
) -> list[tuple[list[int], float]]:
    if batch["mz"].shape[0] != 1:
        raise ValueError("beam_generate currently expects batch size 1")
    memory, mask, fp_logits = model.encode(batch)
    beams = [([tokenizer.bos_id], 0.0, False)]
    for _ in range(max_length - 1):
        expanded = []
        for ids, score, ended in beams:
            if ended:
                expanded.append((ids, score, True))
                continue
            x = torch.tensor([ids], device=memory.device, dtype=torch.long)
            logits = model.decode_logits(memory, mask, x)[:, -1]
            logp = F.log_softmax(logits, dim=-1).squeeze(0)
            values, indices = torch.topk(logp, k=min(beam_size, logp.numel()))
            for v, idx in zip(values.tolist(), indices.tolist()):
                expanded.append((ids + [idx], score + float(v), idx == tokenizer.eos_id))
        expanded.sort(key=lambda z: z[1] / max(1, len(z[0]) - 1), reverse=True)
        beams = expanded[:beam_size]
        if all(b[2] for b in beams):
            break
    return [(ids, score / max(1, len(ids) - 1)) for ids, score, _ in beams]


@torch.no_grad()
def generate_and_rank(
    model,
    batch: dict[str, torch.Tensor | str],
    tokenizer: SmilesTokenizer,
    beam_size: int = 20,
    max_length: int = 192,
    top_k: int = 10,
    mass_tolerance_ppm: float = 10.0,
    require_formula_match_when_available: bool = False,
    sequence_weight: float = 1.0,
    fingerprint_weight: float = 1.5,
    mass_weight: float = 0.25,
) -> list[Candidate]:
    tensor_batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
    memory, mask, fp_logits = model.encode(tensor_batch)
    predicted_fp = torch.sigmoid(fp_logits[0]).detach().cpu().numpy()
    beams = beam_generate(model, tensor_batch, tokenizer, beam_size, max_length)
    precursor = float(tensor_batch["precursor_mz"][0].item())
    adduct = batch.get("adduct", "<unk>")
    if isinstance(adduct, (list, tuple)):
        adduct = adduct[0]
    formula = batch.get("formula", "")
    if isinstance(formula, (list, tuple)):
        formula = formula[0]
    neutral = neutral_mass_from_precursor(precursor, str(adduct))
    out: dict[str, Candidate] = {}
    for ids, seq_score in beams:
        raw = tokenizer.decode(ids)
        can = canonical_smiles(raw)
        if can is None:
            continue
        cand_mass = exact_mass(can)
        mass_ppm = ppm_error(neutral, cand_mass) if neutral is not None and cand_mass is not None else None
        if mass_ppm is not None and abs(mass_ppm) > max(500.0, mass_tolerance_ppm * 20):
            continue
        cand_formula = molecular_formula(can)
        fm = (cand_formula == formula) if formula else None
        if require_formula_match_when_available and formula and not fm:
            continue
        fp = morgan_array(can, len(predicted_fp))
        denom = predicted_fp.sum() + fp.sum() - (predicted_fp * fp).sum() + 1e-8
        fp_score = float((predicted_fp * fp).sum() / denom)
        mass_penalty = 0.0 if mass_ppm is None else min(abs(mass_ppm) / max(mass_tolerance_ppm, 1e-6), 20.0)
        total = sequence_weight * seq_score + fingerprint_weight * fp_score - mass_weight * mass_penalty
        c = Candidate(raw, can, seq_score, fp_score, mass_ppm, fm, total)
        if can not in out or c.total_score > out[can].total_score:
            out[can] = c
    return sorted(out.values(), key=lambda c: c.total_score, reverse=True)[:top_k]
