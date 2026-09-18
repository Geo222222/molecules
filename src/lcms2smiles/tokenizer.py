from __future__ import annotations

from dataclasses import dataclass


SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]
# Fixed lexical vocabulary avoids test-set vocabulary leakage. Multi-character atoms are listed
# before one-character tokens. Bracket expressions are decomposed into these lexical units.
TOKENS = [
    "Cl", "Br", "Si", "Se", "Na", "Li", "Ca", "Mg", "Al", "Fe", "Zn", "Cu", "Mn", "Co",
    "Ni", "As", "Ag", "Sn", "Hg", "Pb", "Bi", "Au", "Pt", "Pd",
    "B", "C", "N", "O", "P", "S", "F", "I", "H", "c", "n", "o", "p", "s", "b",
    "(", ")", "[", "]", "=", "#", "-", "+", "@", "/", "\\", ".", ":", "%",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
]


@dataclass(frozen=True)
class SmilesTokenizer:
    vocab: tuple[str, ...] = tuple(SPECIAL + TOKENS)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stoi", {t: i for i, t in enumerate(self.vocab)})
        object.__setattr__(self, "itos", {i: t for i, t in enumerate(self.vocab)})

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.stoi["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.stoi["<unk>"]

    def lexicalize(self, smiles: str) -> list[str]:
        out: list[str] = []
        i = 0
        multi = sorted((t for t in TOKENS if len(t) > 1), key=len, reverse=True)
        while i < len(smiles):
            hit = next((t for t in multi if smiles.startswith(t, i)), None)
            if hit:
                out.append(hit)
                i += len(hit)
            else:
                out.append(smiles[i])
                i += 1
        return out

    def encode(self, smiles: str, max_length: int | None = None) -> list[int]:
        ids = [self.bos_id]
        ids.extend(self.stoi.get(t, self.unk_id) for t in self.lexicalize(smiles))
        ids.append(self.eos_id)
        if max_length is not None:
            ids = ids[:max_length]
            if ids[-1] != self.eos_id:
                ids[-1] = self.eos_id
            ids += [self.pad_id] * (max_length - len(ids))
        return ids

    def decode(self, ids: list[int]) -> str:
        toks: list[str] = []
        for idx in ids:
            if idx == self.eos_id:
                break
            if idx in (self.pad_id, self.bos_id):
                continue
            tok = self.itos.get(int(idx), "")
            if not tok.startswith("<"):
                toks.append(tok)
        return "".join(toks)
