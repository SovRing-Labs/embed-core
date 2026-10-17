import os, subprocess, sys, tempfile
with tempfile.TemporaryDirectory(prefix='embed-cli-') as t:
 env={**os.environ, 'EMBED_CORE_WEIGHTS': t+'/missing'}
 r=subprocess.run([sys.executable,'-B','bin/embed-core','--help'],capture_output=True,text=True,env=env)
 assert r.returncode==0 and 'embed' in r.stdout and 'armilla' in r.stdout, r.stderr
 r=subprocess.run([sys.executable,'-B','bin/embed-core','embed','public synthetic example'],capture_output=True,text=True,env=env)
 assert r.returncode!=0 and 'Weights directory not found' in r.stderr and t+'/missing' in r.stderr, r.stderr
 print('CLI PASS: parser works; absent external weights reported at configured path')
