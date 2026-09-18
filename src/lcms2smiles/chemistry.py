from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors

PROTON = 1.007276466621
ADDUCT_MASS = {
    "[M+H]+": PROTON,
    "[M-H]-": -PROTON,
    "[M+Na]+": 22.989218,
    "[M+K]+": 38.963158,
    "[M+NH4]+": 18.033823,
}
ELEMENTS = ("C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I", "Si", "B", "Se")


def canonical_smiles(smiles: str, isomeric: bool = True) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)
    except Exception:
        return None


def molecular_formula(smiles: str) -> Optional[str]:
    can = canonical_smiles(smiles)
    if can is None:
        return None
    return rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(can))


def exact_mass(smiles: str) -> Optional[float]:
    can = canonical_smiles(smiles)
    if can is None:
        return None
    return float(Descriptors.ExactMolWt(Chem.MolFromSmiles(can)))


def neutral_mass_from_precursor(precursor_mz: float, adduct: str) -> Optional[float]:
    if adduct not in ADDUCT_MASS:
        return None
    return float(precursor_mz - ADDUCT_MASS[adduct])


def ppm_error(observed_neutral_mass: float, candidate_mass: float) -> float:
    return 1e6 * (candidate_mass - observed_neutral_mass) / observed_neutral_mass


def formula_vector(formula: str | None) -> np.ndarray:
    out = np.zeros(len(ELEMENTS), dtype=np.float32)
    if not formula:
        return out
    counts = {el: int(n) if n else 1 for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula)}
    for i, el in enumerate(ELEMENTS):
        out[i] = counts.get(el, 0)
    return np.log1p(out)


_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)


def morgan_array(smiles: str, n_bits: int = 512) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    arr = np.zeros(n_bits, dtype=np.float32)
    if mol is None:
        return arr
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits).GetFingerprint(mol)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def tanimoto(a: str, b: str) -> float:
    ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
    if ma is None or mb is None:
        return 0.0
    gen = _MORGAN
    return float(DataStructs.TanimotoSimilarity(gen.GetFingerprint(ma), gen.GetFingerprint(mb)))


@dataclass(frozen=True)
class CandidateChemistry:
    smiles: str
    canonical: Optional[str]
    valid: bool
    formula: Optional[str]
    exact_mass: Optional[float]
    mass_error_ppm: Optional[float]
