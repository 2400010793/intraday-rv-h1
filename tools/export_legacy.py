"""One-time migration boundary. Run with --source pointing to the research checkout.
Only this tool imports legacy experiment code; installed inference never does.
"""
import argparse, dataclasses, hashlib, json, pickle, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch


def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,ensure_ascii=False,default=str))


def export(source, output, symbols, months):
    sys.path.insert(0,str(source))  # explicit one-time external-source boundary
    from volatility_forecast.rv_predict.experiments.notebook10_attention import monthly_demo_training as c
    from intraday_rv_h1.models.loader import build_model
    run=source/'results/intraday_rv/notebook10-onds-three-origin-monthly-adam-shampoo-e6-20261014-v1'
    arm='onds15_30_three__drop_negative80'
    m=json.loads((run/'manifest.json').read_text()); seeds=m['seeds']
    if output.exists() and any(output.iterdir()):raise FileExistsError('export needs an empty immutable output directory')
    output.mkdir(parents=True,exist_ok=True)
    ledger=[]
    for symbol in symbols:
        dest=output/symbol;dest.mkdir(exist_ok=True)
        _,_,p=c.original.base.common.c.old.reference_parameters(c.original.base.common.SCOPE,symbol)
        transform=json.loads(Path(p['audit']).read_text())['transform'];write(dest/'transform.json',transform)
        regpath=c.original.base.common.e.d.BASE/'cache/train_plus_validation'/symbol/'registries.pkl'
        with open(regpath,'rb') as f:regs=pickle.load(f)
        np.savez_compressed(dest/'aux_ecdf.npz',**{f'{j}__{clock}':v for j,r in enumerate(regs) for clock,v in r.values.items()})
        # These are fixed mappings/parameters only, not prediction-day observations.
        cache=c.CONTROL/'source_cache'/symbol/'fit';meta=pd.read_parquet(cache/'meta.parquet')
        arrays=[np.load(cache/f'arg{j}.npy',mmap_mode='r') for j in range(10)]
        mapping={}
        for clock,ix in meta.groupby('clock_slot_id',sort=False).indices.items():
            i=int(ix[0]);mapping[clock]={
                'time':arrays[2][i,:3].tolist(),'context':arrays[4][i].tolist(),
                'position':float(arrays[6][i]),'groups':int(arrays[7][i]),
                'own_groups':int(arrays[8][i]),'aux_enabled':arrays[3][i,:6,2].astype(bool).tolist()}
        write(dest/'clock_registry.json',mapping)
        model,_,_=c.model_for(run,symbol,months[0],seeds[0],arm,'adam')
        at=model.attention
        fac={k:getattr(model,k).detach().clone() for k in ['rank_values','rank_counts','own_values','own_counts','own_assets','initial_weights']}
        fac.update(config=dataclasses.asdict(at.config),limits=at.asset_limits.detach().clone(),
            descriptors=at.identity_descriptors.detach().clone(),variant=model.reference.variant,
            reference_state=model.reference.state_dict(),asset=c.original.base.common.e.SYMBOLS.index(symbol),
            numeric_mode=at.numeric_mode,split_calendar=at.split_calendar,
            local_amplitude=symbol in ("IF","IH"),return_column=at.return_column)
        torch.save(fac,dest/'factory.pt')
        for month in months:
            folder=dest/month;folder.mkdir(exist_ok=True);src=run/'cache'/symbol/month
            audit=json.loads((src/'audit.json').read_text())
            for old,new in [('normalization.json','policy_normalization.json'),('registries.npz','policy_ecdf.npz'),(f'{arm}_parameters.npz','ridge.npz')]:shutil.copyfile(src/old,folder/new)
            audit['input_hashes']={str(Path(k).relative_to(source)) if Path(k).is_absolute() and Path(k).is_relative_to(source) else Path(k).name:v for k,v in audit.get('input_hashes',{}).items()}
            write(folder/'fit_audit.json',audit)
            entries=[]
            data=None
            for opt,epoch in [('adam',6),('shampoo_adam_graft',5)]:
                for seed in seeds:
                    checkpoint=run/'tasks'/opt/month/f'{symbol}__seed{seed}'/f'checkpoint_{epoch:02d}.pt'
                    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
                    assert (payload['month'],payload['symbol'],payload['seed'],payload['epoch'])==(month,symbol,seed,epoch)
                    target=folder/f'{opt}__seed{seed}.pt';torch.save(payload['model'],target)
                    if month=='2026-08':
                        if data is None:data=c.Inputs(run,symbol,month,arm)
                        model.load_state_dict(payload['model']);model.eval()
                        local=build_model(fac,payload['model'],seed)
                        # One row at several clocks, including morning/afternoon.
                        sample=np.unique(np.linspace(0,len(data.frame)-1,5).astype(int));args=data.batch('score',sample)
                        with torch.no_grad():
                            ref=model(*args)[0].numpy();got=local(*args)[0].numpy()
                        np.testing.assert_allclose(got,ref,rtol=0,atol=1e-8)
                        archive=np.load(checkpoint.parent/'predictions.npy',mmap_mode='r')[epoch,sample]
                        np.testing.assert_allclose(got,archive,rtol=0,atol=1e-8)
                    entries.append({'optimizer':opt,'epoch':epoch,'seed':seed,'path':target.name,'sha256':digest(target),'source_sha256':digest(checkpoint)})
            entry={'symbol':symbol,'month':month,'max_fit_target_datetime':audit['plan']['max_fit_target_datetime'],'weights':entries}
            write(folder/'manifest.json',entry);ledger.append(entry)
        print(symbol,'exported',len(months),'months',flush=True)
    write(output/'manifest.json',{'schema':'h1-portable-v1','protocol_id':'INTRADAY-SSC-20260829',
        'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
        'symbols':symbols,'months':months,'seeds':seeds,'trained_again':False,'entries':ledger,
        'files':{str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file() and p != output/'manifest.json'}})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--symbols',default='IF,IH,IC,IM,159915,588000');p.add_argument('--months',default='2026-08');a=p.parse_args()
    torch.set_num_threads(1);export(a.source.resolve(),a.output.resolve(),a.symbols.split(','),a.months.split(','))
