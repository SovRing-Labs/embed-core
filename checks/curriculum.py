import copy, sys
sys.path.insert(0, 'curriculum')
import gate, lib
import json
b=json.load(open('curriculum/results/base-s1.json'))
assert not gate.judge(b,b,1)['pass']
c=copy.deepcopy(b); c['levels']['L1']['recall']+=0.1
assert gate.judge(b,c,1)['pass']
c['levels']['L0']['recall']-=0.1
v=gate.judge(b,c,1); assert not v['pass'] and v['earlier']['L0']['violates']
assert not gate.judge(b,b,3)['pass']
c=copy.deepcopy(b); del c['levels']['L1']
assert not gate.judge(b,c,1)['pass']
assert lib.group_heldout('public-example') == lib.group_heldout('public-example')
assert lib.row('L1','public-example','q','p',['n'],'synthetic','CC0')['split'] in ('heldout','train')
print('curriculum-gate PASS: rise, no-rise, regression, low-n, missing, deterministic split')
