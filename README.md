# embed-core — Static token-table embedding, ternary readouts and curriculum regression gates.

[![status: pre-alpha](https://img.shields.io/badge/status-pre--alpha%20(under%20development)-orange)](STATUS.md)
[![defensive publication](https://img.shields.io/badge/defensive%20publication-TECHNICAL--DISCLOSURE.md-blue)](TECHNICAL-DISCLOSURE.md)

Status: **under development — pre-alpha** (`STATUS.md`). Defensive publication: `TECHNICAL-DISCLOSURE.md` (dated 2026-10-16; the DOI is
minted by Zenodo on the first GitHub release).


Static token-table embedding, ternary readouts and curriculum regression gates.

Stage: about 70 %, under development; this is code and aggregate research evidence, not a fitted model distribution.

## Run it
```bash
python3 --version
python3 -B checks/curriculum.py
bash install.sh
```

## What it does today
| Test | Measured property | Evidence |
|---|---|---|
| E1 | The existing judge accepts an isolated L1 rise and rejects no-rise, earlier regression, low sample count and missing target. | `checks/curriculum.py` |
| E2 | Historical static E4 hybrid recall is 0.555 versus FTS 0.735; its preregistered gate failed. | `e4/result.json` |
| E3 | Historical E5 hold-out hybrid recall is 0.640 and the preregistered gate failed. | `e5/result-l0.0001-s150-holdout.json` |
| E4 | Historical E9 hybrid recall 0.745 missed the 0.775 gate. | `e9/e9_gate.json` |

## Not shown here
Full model inference, corpus retrieval, training and cross-host reproduction require the external inputs listed in DATA.md. No teacher/model download is performed by the quick check.

## Files
| Path | Purpose |
|---|---|
| `checks/` | Small export harness invoking existing algorithms |
| `results/` | Aggregate historical evidence and current quick log |
| `suite.json` | Runnable, file and external rows |
| `DATA.md` | Exclusions, hashes and rebuild prerequisites |

## Licence
PolyForm Small Business 1.0.0 OR PolyForm Noncommercial 1.0.0 + Free Trial (code); CC BY 4.0 (documents). See LICENSING.md for eligibility and evaluation terms.
