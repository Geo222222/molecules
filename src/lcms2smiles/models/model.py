from __future__ import annotations

import math

import torch
from torch import nn


class FourierMzEmbedding(nn.Module):
    def __init__(self, d_model: int, n_freq: int = 32) -> None:
        super().__init__()
        self.register_buffer("freq", torch.logspace(-2, 2, n_freq))
        self.proj = nn.Linear(n_freq * 2 + 1, d_model)

    def forward(self, mz: torch.Tensor) -> torch.Tensor:
        x = mz.unsqueeze(-1) * self.freq
        feats = torch.cat([mz.unsqueeze(-1) / 1000.0, torch.sin(x), torch.cos(x)], dim=-1)
        return self.proj(feats)


class SinusoidalPosition(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class SpectrumToSmiles(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 256,
        nhead: int = 8,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_peaks: int = 96,
        max_smiles_tokens: int = 192,
        fingerprint_bits: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.mz_embed = FourierMzEmbedding(d_model)
        self.intensity_embed = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.peak_rank = nn.Embedding(max_peaks + 1, d_model)
        self.meta = nn.Sequential(nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.adduct = nn.Embedding(8, d_model)
        self.instrument = nn.Embedding(8, d_model)
        self.formula = nn.Sequential(nn.Linear(13, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=encoder_layers, norm=nn.LayerNorm(d_model))
        self.token = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = SinusoidalPosition(d_model, max_smiles_tokens + 8)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=decoder_layers, norm=nn.LayerNorm(d_model))
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token.weight
        self.fp_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, fingerprint_bits))

    def encode(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mz, inten, mask = batch["mz"], batch["intensity"], batch["peak_mask"]
        bsz, npeaks = mz.shape
        ranks = torch.arange(npeaks, device=mz.device).unsqueeze(0).expand(bsz, -1)
        peaks = self.mz_embed(mz) + self.intensity_embed(inten.unsqueeze(-1)) + self.peak_rank(ranks)
        meta_scalar = torch.stack([batch["precursor_mz"] / 1000.0, batch["collision_energy"] / 100.0], dim=-1)
        context = (
            self.meta(meta_scalar)
            + self.adduct(batch["adduct_id"])
            + self.instrument(batch["instrument_id"])
            + self.formula(batch["formula_vec"])
        )
        peaks = peaks + context.unsqueeze(1)
        memory = self.encoder(peaks, src_key_padding_mask=mask)
        valid = (~mask).float().unsqueeze(-1)
        pooled = (memory * valid).sum(1) / valid.sum(1).clamp_min(1.0)
        fp_logits = self.fp_head(pooled)
        return memory, mask, fp_logits

    def decode_logits(
        self, memory: torch.Tensor, memory_mask: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        x = self.pos(self.token(token_ids) * math.sqrt(self.d_model))
        t = token_ids.size(1)
        causal = torch.triu(torch.ones(t, t, device=token_ids.device, dtype=torch.bool), diagonal=1)
        out = self.decoder(
            x,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=token_ids.eq(self.pad_id),
            memory_key_padding_mask=memory_mask,
        )
        return self.lm_head(out)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        memory, mask, fp_logits = self.encode(batch)
        decoder_in = batch["smiles_ids"][:, :-1]
        return self.decode_logits(memory, mask, decoder_in), fp_logits
