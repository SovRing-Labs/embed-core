> **Status: under development — pre-alpha.** Interfaces, numbers and file layout can change without notice.

# STATUS — embed-core
Stage: about 70 % (owner estimate).

## Works today
- **E1** The existing judge accepts an isolated L1 rise and rejects no-rise, earlier regression, low sample count and missing target. Evidence: `checks/curriculum.py`.
- **E2** Historical static E4 hybrid recall is 0.555 versus FTS 0.735; its preregistered gate failed. Evidence: `e4/result.json`.
- **E3** Historical E5 hold-out hybrid recall is 0.640 and the preregistered gate failed. Evidence: `e5/result-l0.0001-s150-holdout.json`.
- **E4** Historical E9 hybrid recall 0.745 missed the 0.775 gate. Evidence: `e9/e9_gate.json`.

- **E6** CLI parser and missing-model boundary are executable via `checks/cli.py`.

## Not yet tested
Full inference/training and multi-host reproduction; see the external suite row and DATA.md.

## Known failures and honest negatives
E4 0.555 < FTS 0.735; E5 hold-out 0.640 < FTS 0.735; E9 0.745 < 0.775. The CLI historical headline 0.800 is a calibrated, in-sample result, not an independent quality guarantee. No model is bundled. The E8 comment that a mean of recurrent states is permutation-invariant is too strong: recurrent states depend on order.

## Corrections
First export. Historical headline claims are qualified above; aggregate file checks do not reproduce the underlying retrieval run.

## How to re-check
`python3 -B checks/curriculum.py`; `bash install.sh` opens the local suite interface.

## Defensive publication
TECHNICAL-DISCLOSURE.md dated 2026-10-16; DOI pending, minted by Zenodo on the first GitHub release.
