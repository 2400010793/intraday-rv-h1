"""Thin adapters; no model behavior is implemented here."""
import argparse,json
from pathlib import Path
import pandas as pd
from intraday_rv_h1.data.download import download
from intraday_rv_h1.data.schema import COLUMNS
from intraday_rv_h1.state.bootstrap import bootstrap
from intraday_rv_h1.state.storage import save_state
from intraday_rv_h1.inference.engine import load_bundle


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('download');d.add_argument('--output',type=Path,required=True);d.add_argument('--start',required=True);d.add_argument('--end',required=True);d.add_argument('--evaluation',action='store_true')
    b=sub.add_parser('warmup');b.add_argument('--input',type=Path,required=True);b.add_argument('--bundle',type=Path,required=True);b.add_argument('--symbol',required=True);b.add_argument('--before',required=True);b.add_argument('--output',type=Path,required=True)
    for command in ('replay','stream'):
        r=sub.add_parser(command);r.add_argument('--bundle',type=Path,required=True);r.add_argument('--symbol',required=True);r.add_argument('--month',required=True);r.add_argument('--state',type=Path,required=True);r.add_argument('--output-state',type=Path,required=True)
        if command=='replay':r.add_argument('--input',type=Path,required=True);r.add_argument('--output',type=Path,required=True)
    e=sub.add_parser('evaluate');e.add_argument('--predictions',type=Path,required=True);e.add_argument('--targets',type=Path,required=True);e.add_argument('--thresholds',type=Path,required=True);e.add_argument('--columns',default='shampoo_e5,adam_e6,shampoo_e5_half0.5,adam_e6_half0.5');e.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='evaluate':
        from intraday_rv_h1.evaluation.metrics import evaluate
        metrics=evaluate(pd.read_parquet(a.predictions),pd.read_parquet(a.targets),json.loads(a.thresholds.read_text()),prediction_columns=a.columns.split(','))
        a.output.parent.mkdir(parents=True,exist_ok=True);metrics.to_csv(a.output,index=False);return
    if a.command=='download':print(json.dumps(download(a.output,a.start,a.end,evaluation=a.evaluation)));return
    if a.command=='warmup':
        raw=pd.read_parquet(a.input,columns=list(COLUMNS));raw=raw.loc[pd.to_datetime(raw.datetime)<pd.Timestamp(a.before)]
        registry=json.loads((a.bundle/a.symbol/'clock_registry.json').read_text());save_state(a.output,bootstrap(raw,a.symbol,registry,before=a.before));return
    engine=load_bundle(a.bundle,a.symbol,a.month,a.state)
    if a.command=='stream':
        import sys
        for line in sys.stdin:
            if line.strip():print(json.dumps(engine.predict_step(json.loads(line)),allow_nan=False),flush=True)
    else:
        # The driver may hold the input day, but predict_step only receives one row.
        frame=pd.read_parquet(a.input,columns=list(COLUMNS));out=[]
        for row in frame.to_dict('records'):out.append(engine.predict_step(row))
        a.output.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(out).to_parquet(a.output,index=False)
    engine.save_state(a.output_state)

if __name__=='__main__':main()
