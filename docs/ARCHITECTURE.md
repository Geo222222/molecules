# Architecture

[Back to README](../README.md)

## Objective

Predict a ranked distribution of chemically valid 2D structures from centroided LC-MS/MS spectra. The external representation is canonical SMILES; internal evaluation also uses molecular formula, exact mass and Morgan fingerprints.

## Scientific operating modes

**Formula-assisted** supplies the measured/predicted neutral molecular formula to the network and may use it during reranking. This is useful when formula inference is trusted, but it must never be reported as formula-free de novo performance.

**Formula-free** zeros formula conditioning and relies on precursor mass, adduct, fragment peaks and learned structural evidence. Training applies stochastic formula dropout so this is a first-class operating mode rather than an afterthought.

## Data lifecycle

Raw MassSpecGym v1.5 `fold` assignments are authoritative. No random resplitting is allowed. Peak filtering is deterministic, top-intensity truncation occurs before m/z sorting, intensities are relative-normalized and power-transformed, and no test-derived vocabulary is created.

## Model

The spectrum encoder combines Fourier m/z features, intensity embeddings, peak-rank embeddings and experiment metadata, followed by a Transformer encoder. A pooled spectrum state predicts a Morgan fingerprint as an auxiliary task. An autoregressive Transformer decoder generates SMILES lexical tokens.

## Candidate lifecycle

Generated strings are parsed and canonicalized with RDKit, invalid candidates are removed, duplicate canonical structures are collapsed, neutral exact-mass error is calculated from precursor/adduct information, formula agreement is recorded when available, and candidates are reranked by sequence score, predicted-fingerprint agreement and mass penalty.

## Evaluation contract

Always report exact Top-1/5/10 alongside structural similarity. Report formula-assisted and formula-free results separately. Novelty strata include exact molecule seen in train and Murcko scaffold seen in train. Test data must remain untouched until model and hyperparameters are frozen.

## Upgrade seam

`SpectrumToSmiles.encode()` is the replacement boundary for a DreaMS or other pretrained spectral encoder. Candidate reranking can be upgraded independently to a joint spectrum-molecule model. A future decoding implementation should move precursor-mass constraints into the search itself rather than relying mainly on post-hoc filtering.
