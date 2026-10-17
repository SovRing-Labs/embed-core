# embed-core — technical disclosure

Author: Juan Carlos De Santiago, with the Sovereign Raptor fleet (AI-assisted)
Date: 2026-10-16

## Problem and mechanism
Vocabulary tokens pass through a nomic teacher once. A text is tokenized, its table rows are read, weighted by SIF frequency weights, summed and normalized. Optional PCA reduces width; a deadband maps small coordinates to zero and the rest to signs. E5/E6 fit token tables against teacher document vectors. E8 changes recurrent pooling and E9 trains a reservoir/readout; T1 projects readout features into a ternary target with ridge fitting and straight-through updates. The curriculum groups related examples before a deterministic SHA-256 split, so a group stays on one side. A stage must strictly improve its target level without dropping earlier levels by more than 0.017; fewer than 30 held-out examples cannot certify a level.

## Data layout and rebuilding
Code is organized by experimental stage. JSON files contain aggregate measurements; external arrays store token rows, sparse recurrent weights and dense readouts. DATA.md lists excluded arrays with exact sizes and hashes, dependency layout and regeneration steps. The runnable suite operates only on the shipped code and aggregate values.

## Measured results
| E1 | The existing judge accepts an isolated L1 rise and rejects no-rise, earlier regression, low sample count and missing target. | `checks/curriculum.py` |
| E2 | Historical static E4 hybrid recall is 0.555 versus FTS 0.735; its preregistered gate failed. | `e4/result.json` |
| E3 | Historical E5 hold-out hybrid recall is 0.640 and the preregistered gate failed. | `e5/result-l0.0001-s150-holdout.json` |
| E4 | Historical E9 hybrid recall 0.745 missed the 0.775 gate. | `e9/e9_gate.json` |

## Limits
E4 0.555 < FTS 0.735; E5 hold-out 0.640 < FTS 0.735; E9 0.745 < 0.775. The CLI historical headline 0.800 is a calibrated, in-sample result, not an independent quality guarantee. No model is bundled. The E8 comment that a mean of recurrent states is permutation-invariant is too strong: recurrent states depend on order.

## Field sources
Arora, Liang and Ma, A Simple but Tough-to-Beat Baseline for Sentence Embeddings (2017); Minish Lab, Model2Vec (2024); Jaeger, The Echo State Approach to Analysing and Training Recurrent Neural Networks (2001); Kanerva, Hyperdimensional Computing (2009). These are conceptual sources; no third-party implementation is copied into this export.

Published as defensive prior art: the author claims no patent on what is described here and asks that none be granted on it.
Licence of this text: CC BY 4.0. DOI pending: minted by Zenodo on the first GitHub release.
