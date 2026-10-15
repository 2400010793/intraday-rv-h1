"""Captured-source monthly HARNet scheduler, audited packaging and gh upload."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import numpy as np
import pandas as pd
from .original_monthly_tasks import prepare_task,train_task,write,sha

SYMBOLS=['IF','IH','IC','IM','159915','588000'];SEEDS=[7,17,27,37,47];OPTIMIZERS=['adam','shampoo_adam_graft']


def publish(run):
    """Publish only verified trained assets; raw data/cache/predictions stay local."""
    m=json.loads((run/'manifest.json').read_text());release=run/'release';release.mkdir()
    index=[];allfiles=[];assets=[]
    # Per-symbol archives stay comfortably below GitHub's asset size limit.
    for symbol in SYMBOLS:
        bundle=Path(m['bundle']);mapping={}
        for p in (bundle/symbol).iterdir():
            if p.is_file():mapping[f'{symbol}/{p.name}']=p
        for month in m['months']:
            cache=run/'cache'/month/symbol
            for name in ['shared_initializer.npz','status.json','complete.json']:
                # complete.json references local input arrays: use the Ridge only,
                # and construct a separate public asset index instead.
                if name=='shared_initializer.npz':mapping[f'{symbol}/{month}/initializer.npz']=cache/name
            for p in (bundle/symbol/month).iterdir():
                if p.name in ['policy_normalization.json','policy_ecdf.npz','ridge.npz','fit_audit.json']:mapping[f'{symbol}/{month}/original_{p.name}']=p
            for opt in OPTIMIZERS:
                for seed in SEEDS:
                    task=run/'tasks'/opt/month/f'{symbol}__seed{seed}'
                    status=json.loads((task/'status.json').read_text())
                    if status['status']!='complete' or status['reload_max_error']!=0:raise ValueError('cannot release incomplete/unreplayable weights')
                    # Verify original checkpoint bytes before extracting portable state_dicts.
                    hashes=json.loads((task/'complete.json').read_text())
                    for epoch in range(7):
                        p=task/f'checkpoint_{epoch:02d}.pt'
                        if sha(p)!=hashes[p.name]:raise ValueError('checkpoint changed after training')
                        import torch
                        content=torch.load(p,map_location='cpu',weights_only=False)
                        if (content['symbol'],content['month'],content['seed'],content['epoch'],content['optimizer_name'])!=(symbol,month,seed,epoch,opt):raise ValueError('checkpoint identity mismatch')
                        portable=release/f'{symbol}_{month}_{opt}_{seed}_{epoch:02d}.pt'
                        torch.save(content['model'],portable)
                        rel=f'{symbol}/{month}/{opt}/seed{seed}/epoch_{epoch:02d}.pt';mapping[rel]=portable
                        index.append({'symbol':symbol,'month':month,'optimizer':opt,'seed':seed,'epoch':epoch,'path':rel,'sha256':sha(portable),'max_fit_target_datetime':json.loads((cache/'status.json').read_text())['max_fit_target'],'acceptance':'private research; upstream point-in-time provenance unverified'})
        manifest={'schema':'original-harnet-opening-shared-v1','symbol':symbol,'protocol_id':'INTRADAY-SSC-20260829','acceptance':'research only; not a production leakage certification','inference':'original complete forward graph; same shared weights every clock; use initializer.npz rather than original_ridge.npz; original_ridge.npz retained only as provenance','files':{name:sha(p) for name,p in mapping.items()}}
        meta=release/f'{symbol}_manifest.json';write(meta,manifest)
        archive=release/f'h1-shared-monthly-{symbol}.tar.gz'
        with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
            tar.add(meta,arcname='manifest.json')
            for name,p in sorted(mapping.items()):tar.add(p,arcname=name,recursive=False)
        digest=sha(archive);checksum=archive.with_name(archive.name+'.sha256');checksum.write_text(digest+'  '+archive.name+'\n')
        assets.extend([archive,checksum]);allfiles.append({'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest})
        # Retain extracted state_dict files for local reproducibility, not raw states.
    write(release/'model_index.json',index);write(release/'archive_index.json',allfiles);assets.extend([release/'model_index.json',release/'archive_index.json'])
    notes=release/'notes.txt';notes.write_text('Original complete HARNet, six independent assets, all clocks shared including causal auction inputs. Twenty monthly mature-history fits; 5 fixed seeds; Adam and Shampoo6epochs; epoch0..6 retained. Upstream publication/revision provenance remains unverified: private research draft, no production certification. Raw market observations, historical streaming states, targets and predictions are excluded. Source/input/asset hashes recorded locally.\n')
    repo='2400010793/intraday-rv-h1';tag=m['release_tag']
    privacy=json.loads(subprocess.check_output(['gh','repo','view',repo,'--json','visibility'],text=True))
    if privacy['visibility']!='PRIVATE':raise ValueError('refuse publishing to nonprivate repo')
    subprocess.run(['gh','release','create',tag,'--repo',repo,'--draft','--title','H1 original HARNet — monthly shared opening weights','--notes-file',str(notes)],check=True)
    for asset in assets:subprocess.run(['gh','release','upload',tag,str(asset),'--repo',repo],check=True)
    result=json.loads(subprocess.check_output(['gh','release','view',tag,'--repo',repo,'--json','assets,isDraft,url'],text=True))
    remote={v['name']:v['size'] for v in result['assets']}
    for p in assets:
        if remote.get(p.name)!=p.stat().st_size:raise ValueError('Release asset size verification failed')
    write(release/'upload_acceptance.json',result)
    return result


def pipeline(run):
    run=Path(run).resolve();m=json.loads((run/'manifest.json').read_text());start=time.monotonic()
    if sha(Path(__file__))!=m['pipeline_sha256']:raise ValueError('use captured pipeline')
    ctx=multiprocessing.get_context('spawn')
    try:
        m.update(status='running',stage='preflight');write(run/'manifest.json',m)
        # Representative early/latest months, future-label and closure gates before fanout.
        cache_tests=[('IF','2025-03'),('159915','2026-09')]
        with ProcessPoolExecutor(max_workers=2,mp_context=ctx) as pool:
            records=list(pool.map(prepare_task,[run]*2,[p[0] for p in cache_tests],[p[1] for p in cache_tests]))
        tasks=[(s,mo,opt) for s,mo in cache_tests for opt in OPTIMIZERS]
        with ProcessPoolExecutor(max_workers=4,mp_context=ctx) as pool:
            fs=[pool.submit(train_task,run,s,mo,7,opt,16) for s,mo,opt in tasks];smoke=[f.result() for f in fs]
        # Measured cost projection. Includes four workers' full historical loading/scoring.
        seconds_per_update=max(s['train_seconds']/s['updates'] for s in smoke)
        total_updates=sum(6*((r['regular_fit_rows']+r['opening_fit_rows']+127)//128) for r in records)/2*1200
        projected=seconds_per_update*total_updates/60
        preflight={'status':'passed','cache':records,'optimizer_smoke':smoke,'measured_seconds_per_update_max':seconds_per_update,'estimated_training_seconds_60workers':projected,'peak_rss_kib':max(s['peak_rss_kib'] for s in smoke)}
        write(run/'preflight.json',preflight)
        if projected>m['hard_timeout_seconds']*.85:raise RuntimeError(f'preflight projects {projected}s above runtime budget')
        m.update(stage='cache',preflight_status='passed',projected_training_seconds=projected);write(run/'manifest.json',m)
        with ProcessPoolExecutor(max_workers=60,mp_context=ctx) as pool:
            fs=[pool.submit(prepare_task,run,s,mo) for mo in m['months'] for s in SYMBOLS]
            for f in as_completed(fs):f.result()
        m.update(stage='train',training_started_at=time.time());write(run/'manifest.json',m)
        # Independent30-worker pools, both use same fitted initialization/shuffle.
        with ProcessPoolExecutor(max_workers=30,mp_context=ctx) as adam,ProcessPoolExecutor(max_workers=30,mp_context=ctx) as shampoo:
            fs=[]
            for opt,pool in [('adam',adam),('shampoo_adam_graft',shampoo)]:
                for mo in m['months']:
                    for s in SYMBOLS:
                        for seed in SEEDS:fs.append(pool.submit(train_task,run,s,mo,seed,opt))
            count=0
            for f in as_completed(fs):
                f.result();count+=1;write(run/'progress.json',{'status':'running','completed_trajectories':count,'trajectories':1200,'seconds':time.monotonic()-start})
        m.update(stage='package');write(run/'manifest.json',m)
        uploaded=publish(run)
        m.update(status='complete',stage='complete',seconds=time.monotonic()-start,release_url=uploaded['url'],release_draft=uploaded['isDraft']);write(run/'manifest.json',m)
    except BaseException as exc:
        m.update(status='failed',error_type=type(exc).__name__,error=str(exc),seconds=time.monotonic()-start);write(run/'manifest.json',m);raise


def prepare(a):
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=False);local=Path(__file__).resolve().parents[3];source=Path(a.source).resolve()
    capture=root/'source_snapshot';capture.mkdir();shutil.copytree(local/'src',capture/'src',ignore=shutil.ignore_patterns('__pycache__'));shutil.copytree(source/'volatility_forecast',capture/'legacy/volatility_forecast',ignore=shutil.ignore_patterns('__pycache__'));shutil.copy2(source/'pyproject.toml',capture/'legacy/pyproject.toml')
    original=json.loads((source/'results/intraday_rv/notebook10-onds-three-origin-monthly-adam-shampoo-e6-20261014-v1/manifest.json').read_text())
    months=original['months'];assert len(months)==20
    m={'status':'prepared','stage':'prepared','protocol_id':'INTRADAY-SSC-20260829','source':str(source),'bundle':str(Path(a.bundle).resolve()),'months':months,'symbols':SYMBOLS,'seeds':SEEDS,'optimizers':OPTIMIZERS,'epochs':6,'workers':60,'worker_pools':{'adam':30,'shampoo_adam_graft':30},'threads':1,'trajectories':1200,'target':'H1 next-session same physical clock','model':'original complete MonthlyDemoHARNet forward and loss; original convolution trainability; original L preserved','fit_scope':'monthly original mature history plus causal opening samples shared with all clocks; original frozen transforms retained; refit original Ridge with opening','score_scope':'complete sessions after each original cutoff, new auction before cutoff explicitly excluded','selection':'fixed epochs and all5seeds; Test diagnostic only; earlier Test informed followup','prior_protocol':'six full epochs both optimizers, Adam lr1e-5 batch128; Shampoo original attention partition/Adam graft roots every10; fresh epoch shuffles same as original','transforms':'original month-frozen ECDF/scalers/source tensors; no Test refitting; candidate Ridge refit includes auctions mapped to first trained clock','hard_timeout_seconds':48*3600,'expected_runtime':'preliminary8..24hours, replaced by preflight measured projection','memory_budget_bytes':480*1024**3,'tmux':'h1-original-monthly-60-20261015','cancellation':'tmux send-keys -t h1-original-monthly-60-20261015 C-c; check all descendants; preserve artifacts','release_repo':'2400010793/intraday-rv-h1','release_tag':'h1-shared-monthly-20261015-v1','release_policy':'automatic private draft Release after all tasks and replay/hash gates pass; no raw data or history states','source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),'source_hashes':{str(p.relative_to(capture)):sha(p) for p in capture.rglob('*.py')},'pipeline_sha256':sha(capture/'src/intraday_rv_h1/research/original_monthly_pipeline.py'),'input_hashes':{str(p):sha(p) for p in [source/'results/intraday_rv/notebook10-onds-three-origin-monthly-adam-shampoo-e6-20261014-v1/manifest.json',Path(a.bundle).resolve()/'manifest.json']}}
    write(root/'manifest.json',m);print(root)

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True);q=sub.add_parser('prepare');q.add_argument('--source',required=True);q.add_argument('--bundle',required=True);q.add_argument('--output',required=True);q=sub.add_parser('run');q.add_argument('--run',required=True);a=p.parse_args();prepare(a) if a.command=='prepare' else pipeline(a.run)
