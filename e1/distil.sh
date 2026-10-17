#!/bin/bash
# EMBED-CORE-0 E1: every nomic vocabulary token through the model once (raw, mean pooling, L2-normalised) — cos 1.0000 vs omnid (10-08 check).
set -euo pipefail
E=$HOME/projects/bitnet/build/bin/llama-embedding; M=$HOME/projects/bitnet/models/nomic-embed-text-v1.5/nomic-embed-1.5-f16.gguf
V=$HOME/projects/bitnet/models/nomic-embed-text-v1.5/vocab.txt; O=$(dirname "$0")/vocab; mkdir -p $O
split -l 2048 -d -a 3 $V $O/part.
for p in $O/part.*; do
  [ -s $p.emb ] && continue
  timeout 600 $E -m $M -f $p --pooling mean --embd-normalize 2 --embd-output-format array -b 16384 -ub 16384 -t 6 > $p.emb 2> $p.log
done
echo "distil done: $(ls $O/part.*.emb | wc -l) parts"
