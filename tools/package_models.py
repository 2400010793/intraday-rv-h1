"""Verify all assets and build a private research-only Release archive."""
import argparse,hashlib,json,tarfile
from pathlib import Path


def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 m=json.loads((a.bundle/'manifest.json').read_text());actual={str(x.relative_to(a.bundle)) for x in a.bundle.rglob('*') if x.is_file() and x!=a.bundle/'manifest.json'}
 if actual!=set(m['files']):raise ValueError('manifest coverage differs from actual files')
 for key,digest in m['files'].items():
  if sha(a.bundle/key)!=digest:raise ValueError(f'hash mismatch: {key}')
 if a.output.exists():raise FileExistsError(a.output)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with tarfile.open(a.output,'w:gz',compresslevel=1) as tar:
  for path in sorted(a.bundle.rglob('*')):
   if path.is_file():tar.add(path,arcname=str(path.relative_to(a.bundle)),recursive=False)
 result=sha(a.output);a.output.with_name(a.output.name+'.sha256').write_text(result+'  '+a.output.name+'\n')
 print(json.dumps({'file':str(a.output),'bytes':a.output.stat().st_size,'sha256':result,'files':len(actual)+1,'acceptance':m['acceptance']},indent=2))
