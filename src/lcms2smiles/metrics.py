from __future__ import annotations

from dataclasses import dataclass, asdict

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .chemistry import canonical_smiles, tanimoto


def scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol), canonical=True)


@dataclass
class EvalRecord:
    exact_top1: int
    exact_top5: int
    exact_top10: int
    best_tanimoto: float
    top1_tanimoto: float
    valid_candidates: int
    target_seen_in_train: int
    scaffold_seen_in_train: int

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_candidates(target: str, candidates: list[str], train_smiles: set[str], train_scaffolds: set[str]) -> EvalRecord:
    target_can = canonical_smiles(target) or target
    cans = [canonical_smiles(s) for s in candidates]
    cans = [s for s in cans if s]
    matches = [int(s == target_can) for s in cans]
    sims = [tanimoto(target_can, s) for s in cans]
    return EvalRecord(
        exact_top1=int(bool(matches[:1] and matches[0])),
        exact_top5=int(any(matches[:5])),
        exact_top10=int(any(matches[:10])),
        best_tanimoto=max(sims, default=0.0),
        top1_tanimoto=sims[0] if sims else 0.0,
        valid_candidates=len(cans),
        target_seen_in_train=int(target_can in train_smiles),
        scaffold_seen_in_train=int(scaffold(target_can) in train_scaffolds),
    )
