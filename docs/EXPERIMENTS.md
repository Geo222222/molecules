# Experiment protocol

[Back to README](../README.md) · [Architecture](ARCHITECTURE.md)

Every run should record: git commit, config file, dataset version/checksum, operating mode (formula-assisted or formula-free), random seed, GPU model, training time, best validation epoch, and untouched test metrics.

The primary scorecard is exact Top-1/5/10 plus best/top-1 Morgan Tanimoto. Secondary diagnostics are valid candidate count, precursor-mass ppm error distribution, formula consistency, exact train-molecule overlap and Murcko-scaffold overlap.

A result should not be called a novel-structure result merely because the exact SMILES was absent from training. The stronger claim requires scaffold-disjoint or benchmark-defined leakage-aware evaluation. When orthogonal evidence is available, retain it as metadata rather than silently mixing it into an MS/MS-only benchmark.
