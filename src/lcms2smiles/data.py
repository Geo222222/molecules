from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .chemistry import formula_vector, morgan_array
from .tokenizer import SmilesTokenizer

ADDUCTS = {"<unk>": 0, "[M+H]+": 1, "[M-H]-": 2, "[M+Na]+": 3, "[M+K]+": 4, "[M+NH4]+": 5}
INSTRUMENTS = {"<unk>": 0, "Orbitrap": 1, "QTOF": 2, "TOF": 3, "FTICR": 4}


@dataclass
class SpectrumExample:
    mz: np.ndarray
    intensity: np.ndarray
    precursor_mz: float
    adduct: str
    instrument_type: str
    collision_energy: float
    formula: str | None
    smiles: str
    identifier: str


def preprocess_peaks(
    mz: np.ndarray,
    intensity: np.ndarray,
    max_peaks: int,
    min_relative_intensity: float = 0.001,
    intensity_power: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mz = np.asarray(mz, dtype=np.float32)
    intensity = np.asarray(intensity, dtype=np.float32)
    if len(mz) != len(intensity):
        raise ValueError("m/z and intensity arrays must have the same length")
    if len(mz) == 0:
        return np.zeros(max_peaks, np.float32), np.zeros(max_peaks, np.float32), np.ones(max_peaks, bool)
    intensity = np.maximum(intensity, 0)
    mx = float(intensity.max()) or 1.0
    intensity = intensity / mx
    keep = intensity >= min_relative_intensity
    mz, intensity = mz[keep], intensity[keep]
    if len(mz) > max_peaks:
        idx = np.argpartition(intensity, -max_peaks)[-max_peaks:]
        mz, intensity = mz[idx], intensity[idx]
    order = np.argsort(mz)
    mz, intensity = mz[order], intensity[order]
    intensity = np.power(intensity, intensity_power)
    pad = max_peaks - len(mz)
    mask = np.array([False] * len(mz) + [True] * pad, dtype=bool)
    return (
        np.pad(mz, (0, pad)).astype(np.float32),
        np.pad(intensity, (0, pad)).astype(np.float32),
        mask,
    )


class MassSpecGymDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        fold: str,
        tokenizer: SmilesTokenizer,
        max_peaks: int = 96,
        max_smiles_tokens: int = 192,
        min_relative_intensity: float = 0.001,
        intensity_power: float = 0.5,
        fingerprint_bits: int = 512,
    ) -> None:
        self.path = Path(path)
        df = pd.read_csv(self.path, sep="\t")
        if "fold" not in df.columns:
            raise ValueError("Expected MassSpecGym-compatible TSV with a fold column")
        self.df = df[df["fold"] == fold].reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_peaks = max_peaks
        self.max_smiles_tokens = max_smiles_tokens
        self.min_relative_intensity = min_relative_intensity
        self.intensity_power = intensity_power
        self.fingerprint_bits = fingerprint_bits

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _csv_floats(value: str) -> np.ndarray:
        if pd.isna(value) or not str(value).strip():
            return np.array([], dtype=np.float32)
        return np.fromstring(str(value), sep=",", dtype=np.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        r = self.df.iloc[idx]
        mz, intensity, peak_mask = preprocess_peaks(
            self._csv_floats(r["mzs"]),
            self._csv_floats(r["intensities"]),
            self.max_peaks,
            self.min_relative_intensity,
            self.intensity_power,
        )
        formula = None if pd.isna(r.get("formula")) else str(r.get("formula"))
        ce = 0.0 if pd.isna(r.get("collision_energy")) else float(r.get("collision_energy"))
        inst = str(r.get("instrument_type", "<unk>"))
        adduct = str(r.get("adduct", "<unk>"))
        smiles = str(r["smiles"])
        return {
            "mz": torch.from_numpy(mz),
            "intensity": torch.from_numpy(intensity),
            "peak_mask": torch.from_numpy(peak_mask),
            "precursor_mz": torch.tensor(float(r["precursor_mz"]), dtype=torch.float32),
            "collision_energy": torch.tensor(ce, dtype=torch.float32),
            "adduct_id": torch.tensor(ADDUCTS.get(adduct, 0), dtype=torch.long),
            "instrument_id": torch.tensor(INSTRUMENTS.get(inst, 0), dtype=torch.long),
            "formula_vec": torch.from_numpy(formula_vector(formula)),
            "smiles_ids": torch.tensor(self.tokenizer.encode(smiles, self.max_smiles_tokens), dtype=torch.long),
            "fingerprint": torch.from_numpy(morgan_array(smiles, self.fingerprint_bits)),
            "smiles": smiles,
            "formula": formula or "",
            "adduct": adduct,
            "identifier": str(r.get("identifier", idx)),
        }
