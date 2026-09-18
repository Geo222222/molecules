# LCMS2SMILES

A leakage-aware machine-learning environment for generating ranked 2D molecular-structure candidates (canonical SMILES) from high-resolution LC-MS/MS spectra.

## What this repository is

This is a research-grade baseline environment, not a claim that MS/MS uniquely determines every 2D structure. The model generates multiple candidates and ranks them using sequence likelihood, a learned spectral fingerprint, precursor-mass agreement, and optional formula agreement. Evaluation explicitly separates exact-match performance from structural similarity and train-set/scaffold novelty.

## Documentation

- [Architecture and scientific modes](docs/ARCHITECTURE.md)
- [Experiment and reporting protocol](docs/EXPERIMENTS.md)

## Architecture

1. **Input contract** — centroided MS/MS peaks, precursor m/z, adduct, collision energy, instrument type, and optional formula.
2. **Peak encoder** — continuous Fourier m/z features + intensity embeddings + peak-rank embeddings + experimental metadata, processed by a Transformer encoder.
3. **Multi-task structural head** — predicts a Morgan/ECFP-like structural fingerprint from the spectrum representation.
4. **SMILES decoder** — autoregressive Transformer trained with teacher forcing over a fixed lexical SMILES vocabulary.
5. **Chemistry-aware candidate ranking** — canonicalization/validity via RDKit, predicted-fingerprint agreement, precursor neutral-mass error, optional molecular-formula constraint, candidate deduplication.
6. **Leakage-aware evaluation** — Top-1/5/10 exact structure, Tanimoto similarity, candidate validity, exact train-molecule overlap, and Murcko-scaffold overlap.

The encoder is intentionally modular so a DreaMS-derived pretrained spectral encoder can replace the native encoder later without changing the generation/evaluation contract.

## Benchmark

The default config targets **MassSpecGym v1.5**. Keep the official `fold` values intact; do not randomly re-split spectra. The dataset contains repeated spectra for some molecules, so a spectrum-level random split can produce severe leakage.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e '.[dev]'
make test
```

Download MassSpecGym v1.5:

```bash
make data
```

Train:

```bash
make train
```

Evaluate the untouched test fold:

```bash
python -m lcms2smiles.evaluate \
  --config configs/base.yaml \
  --checkpoint artifacts/checkpoints/best.pt \
  --fold test \
  --output artifacts/test-evaluation.json
```

Inspect one benchmark prediction:

```bash
python -m lcms2smiles.predict \
  --config configs/base.yaml \
  --checkpoint artifacts/checkpoints/best.pt \
  --fold test \
  --index 0
```

## Definition of done for the baseline

The baseline is considered experimentally operational when: (1) CI/tests pass; (2) MassSpecGym v1.5 downloads reproducibly; (3) a training run creates `best.pt` and `last.pt`; (4) test evaluation reports exact Top-1/5/10, fingerprint similarity and novelty strata; (5) prediction emits ranked, valid, deduplicated canonical SMILES with mass/formula diagnostics; and (6) no test spectra or test molecules are used to build vocabularies or tune model parameters.

## Next research upgrades

The strongest next steps are: DreaMS encoder initialization, precursor-mass-shell constrained decoding rather than post-hoc filtering, formula-prediction uncertainty instead of oracle formula conditioning, self-supervised pretraining on millions of unlabeled spectra, synthetic-spectrum transfer learning, and a learned spectrum↔molecule joint-embedding reranker. For novel-compound claims, report the exact data split and distinguish formula-known from formula-free inference.

## Important scientific limitation

High-resolution LC-MS/MS does not always contain enough information to distinguish constitutional isomers or stereoisomers. Therefore a calibrated candidate distribution is scientifically preferable to a single unqualified structure prediction. Orthogonal evidence (retention time, ion mobility/CCS, NMR, standards, or additional MSn) should be incorporated when available.

Run inference on your own JSON spectrum:

```bash
python -m lcms2smiles.inference \
  --config configs/base.yaml \
  --checkpoint artifacts/checkpoints/best.pt \
  --input examples/spectrum.json
```

For a realistic unknown where the molecular formula is not available, omit `formula` from the JSON. Benchmark both regimes explicitly; use `--formula-free` with `predict` or `evaluate` to suppress formula conditioning even when the benchmark row contains it.
