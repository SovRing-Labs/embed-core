# AGENTS.md — embed-core

For coding assistants (and people) setting this repository up. Plain steps; every command runs from the repo root.

**Status: pre-alpha, under development.** Interfaces, numbers and file layout can change without notice. Read `STATUS.md` first: what works today (each line names its test and evidence file), what is not yet tested, the known failures and the corrections.

## What it is
Stage: about 70 % (under development, pre-alpha).
Static token-table embedding, ternary readouts and curriculum regression gates.

## Install and check (3 commands)
```bash
git clone --depth 1 https://github.com/SovRing-Labs/embed-core && cd embed-core
# nothing to install beyond: python3, numpy and scipy for the CLI boundary check
python3 -B checks/curriculum.py   # expect: PASS
```
Needs: python3, numpy and scipy for the CLI boundary check.
Whole suite instead: `bash install.sh --no-gui` (python check → venv → every runnable `suite.json` test, headless; without `--no-gui` it then opens the local results page on 127.0.0.1).

GUI: `bash install.sh`

## What the tests mean
`suite.json` lists every claim this repository makes with its verify command, the expected output, what passing means, how the number was obtained and how to use it in your own work (6 tests, in `suite.json` at the repository root). `STATUS.md` says what works today, what is not yet tested and the known failures; the umbrella's `STATUS.md` has one row per repository.

## Where the evidence is
- `README.md`
- `TECHNICAL-DISCLOSURE.md` — the method, as dated prior art
- `CITATION.cff`

## Honest limits (from STATUS.md)
E4 0.555 < FTS 0.735; E5 hold-out 0.640 < FTS 0.735; E9 0.745 < 0.775. The CLI historical headline 0.800 is a calibrated, in-sample result, not an independent quality guarantee. No model is bundled. The E8 comment that a mean of recurrent states is permutation-invariant is too strong: recurrent states depend on order.

## Licence
PolyForm Small Business 1.0.0 OR PolyForm Noncommercial 1.0.0 + Free Trial (code); CC BY 4.0 (documents). Details: `LICENSING.md`.

## Guided setup through MCP
The SovRing umbrella ships an MCP server that explains, checks readiness, clones and runs these tests: https://github.com/SovRing-Labs/SovRing (README → "Paste this to your assistant"). Zero-install docs endpoint: https://gitmcp.io/SovRing-Labs/embed-core
