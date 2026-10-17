#!/bin/bash
# Gamma is the ONLY new hyperparameter. Cheap, closed-form refit each time (~3min).
# Recorded before running so the range cannot be moved afterwards.
for g in 0.80 0.90 0.95 0.99 0.995; do
  echo "=== gamma=$g ==="
  python3 -u fit_eval.py $g 2>&1 | python3 -c "
import sys,json
t=sys.stdin.read()
i=t.find('{')
if i<0: print(t[-200:]); raise SystemExit
d=json.loads(t[i:])
print('  core_only=%s hybrid=%s valcos=%s %s'%(d['core_only'],d['hybrid_fts_core'],d['val_cosine_to_nomic'],d['verdict']))"
  cp e8_result.json "res_gamma_${g}.json"
done
echo "SWEEP_DONE"
