"""One production row -> local features -> frozen models. No label access."""
import bisect,copy,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from intraday_rv_h1.data.schema import validate_row, validate_available_row
from intraday_rv_h1.features.native import raw_features,midrank,extended_log
from intraday_rv_h1.models.loader import build_model
from intraday_rv_h1.state.storage import load_state as read_state,save_state as write_state


def _sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


class Predictor:
    def __init__(self,bundle,symbol,month,state,*,optimizers=('shampoo_adam_graft','adam')):
        torch.set_num_threads(1)
        self.root=Path(bundle);self.symbol=symbol;self.month=month
        top=json.loads((self.root/'manifest.json').read_text())
        base=self.root/symbol;folder=base/month
        manifest_key=str((folder/'manifest.json').relative_to(self.root))
        if top['files'].get(manifest_key)!=_sha(folder/'manifest.json'):raise ValueError('model manifest hash mismatch')
        self.manifest=json.loads((folder/'manifest.json').read_text())
        self.version=_sha(folder/'manifest.json')
        self.state=copy.deepcopy(state)
        if state['symbol']!=symbol:raise ValueError('state symbol differs')
        if state.get('weight_version') not in (None,self.version):raise ValueError('state belongs to different weights')
        self.state['weight_version']=self.version
        if pd.Timestamp(self.manifest['max_fit_target_datetime'])>pd.Timestamp(state['last_datetime']):raise ValueError('weights contain labels newer than history')
        self.registry=json.loads((base/'clock_registry.json').read_text());self.transform=json.loads((base/'transform.json').read_text())
        self.normalization=json.loads((folder/'policy_normalization.json').read_text())
        with np.load(base/'aux_ecdf.npz') as z:self.aux_ecdf={k:z[k] for k in z.files}
        with np.load(folder/'policy_ecdf.npz') as z:self.policy_ecdf={k:z[k] for k in z.files}
        # Verify every runtime asset used, including non-persistent rank tables.
        for p in [base/'factory.pt',base/'transform.json',base/'clock_registry.json',base/'aux_ecdf.npz',folder/'policy_normalization.json',folder/'policy_ecdf.npz']:
            key=str(p.relative_to(self.root))
            if top['files'].get(key)!=_sha(p):raise ValueError(f'asset hash mismatch: {key}')
        factory=torch.load(base/'factory.pt',map_location='cpu',weights_only=True)
        self.models={}
        for opt in optimizers:
            self.models[opt]=[]
            for entry in self.manifest['weights']:
                if entry['optimizer']!=opt:continue
                p=folder/entry['path']
                if _sha(p)!=entry['sha256']:raise ValueError(f'weight hash mismatch: {p.name}')
                weights=torch.load(p,map_location='cpu',weights_only=True)
                self.models[opt].append(build_model(factory,weights,entry['seed']))
            if len(self.models[opt])!=len(top['seeds']):raise ValueError('missing fixed seeds')
        self.asset=factory['asset'];self.last_args=None

    def save_state(self,path):write_state(path,self.state)

    def _features(self,s,row):
        dt=pd.Timestamp(row['datetime']);date=dt.normalize();flags=[f'missing_{k}' for k in ('ret','spotRef','rvol1D','rvol1hour') if row[k] is None]
        olddate=pd.Timestamp(s['date'])
        if date>olddate:
            # Daily observations enter expanding CDFs only after their session.
            for name in ('opening','overnight'):
                val=s['today_'+name]
                if val is not None:bisect.insort(s[name+'_history'],val)
            if olddate!=pd.Timestamp(s['previous_session']):
                oldtime=pd.Timestamp(s.get('max_observed_datetime',s['last_datetime']))
                if oldtime.strftime('%H:%M')=='15:00' and s['price_segment_valid']:
                    s['price_closes']=(s['price_closes']+[s['log_price']])[-80:]
            s.update(previous_session=s['date'],date=date.isoformat(),today_opening=None,today_overnight=None,observed_today=[],last_predictions={})
        # clock occurrence is based on received physical records, never rounded.
        today=s['observed_today'];minute=dt.strftime('%H:%M');occ=sum(pd.Timestamp(x['datetime']).strftime('%H:%M')==minute for x in today)
        clock=f'{minute}__{occ}';position=len(today)
        if clock in self.registry:
            mapped=clock
            actual_position=int(self.registry[clock]['position'])
            if actual_position!=position:flags.append('physical_clock_gap')
            position=actual_position
        elif position==0:
            mapped=min(self.registry,key=lambda c:self.registry[c]['position']);flags.append('opening_clock_mapped')
        else:raise ValueError(f'unregistered non-opening physical clock: {clock}')
        template=self.registry[mapped]
        cooked=dict(row,clock_position=position);today.append(cooked);s['raw_rows'].append(cooked)
        # Retain two preceding sessions and today's partial session. Rolling support is bounded.
        dates=sorted({r['trade_date'] for r in s['raw_rows']});keep=set(dates[-3:]);s['raw_rows']=[r for r in s['raw_rows'] if r['trade_date'] in keep]
        raw=pd.DataFrame(s['raw_rows']);features=raw_features(raw,self.symbol);f=features.iloc[-1]
        core=np.array([f.core0,f.core1,f.core2],np.float32)
        original_core_valid=np.isfinite(core).all()
        if position==0:
            # Opening has no intraday return. Use the completed trailing semivariance support.
            interior=features.loc[features.clock_position.gt(0),'ret'].to_numpy(float)
            width=242 if self.asset<4 else 239;rr=interior[-width:]
            neg=260*np.where(rr<0,rr*rr,0.).sum() if np.isfinite(rr).all() else np.nan
            pos=260*np.where(rr>0,rr*rr,0.).sum() if np.isfinite(rr).all() else np.nan
            rv=float(row['rvol1D'])**2 if row['rvol1D'] is not None else np.nan
            with np.errstate(divide='ignore',invalid='ignore'):
                core=np.array([np.log(rv),np.log(neg),np.log(1+(pos-neg)/rv)],np.float32)
            flags.append('opening_semivariance_extension')
        for j,col in [(1,'negative_variance'),(2,'log_argument')]:
            if s['extension_mode']!='carry' and np.isfinite(f[col]) and float(f[col])<s['thresholds'][col]:
                val=extended_log(float(f[col]),s['thresholds'][col],s['extension_mode'])
                if np.isfinite(val):core[j]=val;flags.append('log_'+s['extension_mode'])
        history=s['histories'][mapped]
        for j in range(3):
            if not np.isfinite(core[j]):core[j]=history[-1][j];flags.append(f'core{j}_historical_extension')
        aux=np.zeros(6,np.float64);ranks=np.zeros(6,np.float64);availability=np.array(template['aux_enabled'],bool)
        rawaux=np.array([f[f'aux{j}'] for j in range(2,6)],float)
        for j in range(2,6):
            if not availability[j]:continue
            value=rawaux[j-2]
            if not np.isfinite(value):
                value=s['last_aux'][mapped][j-2];flags.append(f'aux{j}_historical_extension')
                if value is None:raise ValueError(f'no historical auxiliary {mapped}/{j}')
            else:s['last_aux'][mapped][j-2]=float(value)
            # Preserve legacy float32 window materialization before scalar normalization.
            normalized=(np.float32(value)-np.float32(self.transform['input_mean'][j+3]))/np.float32(self.transform['input_std'][j+3])
            aux[j]=float(normalized)
            ranks[j]=2*midrank(self.aux_ecdf[f'{j}__{mapped}'],aux[j])-1
        aux[5]=0.;ranks[5]=0.;availability[5]=False
        triplets=np.column_stack((ranks,np.maximum(ranks-.6,0)/.4,availability.astype(float)))
        state=np.zeros((8,3));state[:6]=triplets;state[6:,2]=1.
        old=np.r_[template['time'],triplets.ravel()]
        policy=np.zeros((2,3))
        for j in range(2):
            v=float(f[f'policy{j}']);ok=np.isfinite(v)
            if ok:
                v=(v-self.normalization[j]['mean'])/self.normalization[j]['std']
                policy[j]=[v,2*midrank(self.policy_ecdf[f'{j}__{mapped}'],v)-1,1.]
            else:flags.append(f'policy{j}_unavailable_masked')
        window=np.array(history[-79:]+[core.tolist()],np.float32)
        mean=np.array(self.transform['input_mean'][:3],np.float32);std=np.array(self.transform['input_std'][:3],np.float32)
        window=(window-mean[None,:])/std[None,:]
        if original_core_valid and clock in self.registry and all(np.isfinite(rawaux[j-2]) for j in range(2,6) if template['aux_enabled'][j]):s['histories'][clock]=(history+[core.tolist()])[-80:]
        # Signed raw return and price corrections.
        r=row['ret'];r=np.nan if r is None else r
        if position==0 and np.isfinite(r):s['today_overnight']=float(r)
        if len(today)>1 and today[0]['clock_position']==0 and today[1]['clock_position']==1 and s['today_opening'] is None:
            first=pd.Timestamp(today[1]['datetime']);elapsed=(dt-first).total_seconds()
            if elapsed>=900:
                support=[x for x in today[2:] if first<pd.Timestamp(x['datetime'])<=first+pd.Timedelta(minutes=15)]
                complete=len(support)>0 and all(x['ret'] is not None for x in support)
                stamp=[pd.Timestamp(today[1]['datetime'])]+[pd.Timestamp(x['datetime']) for x in support]
                complete=complete and all(0<(b-a).total_seconds()<=90 for a,b in zip(stamp,stamp[1:])) and stamp[-1]>=first+pd.Timedelta(minutes=15)
                if complete:s['today_opening']=float(sum(x['ret'] for x in support))
        returns=np.zeros((5,3))
        returns[:,0]=.5
        coefficient=np.zeros((3,3));coefficient[2,0]=.5;coefficient[2,2]=.5
        for name,slot,ci in [('overnight',3,0),('opening',1,1)]:
            value=s['today_'+name]
            if value is not None:
                hist=s[name+'_history'];available=len(hist)>=20
                returns[slot]=[midrank(hist,value) if available else .5,float(available),value]
                coefficient[ci]=[value,1.,0.]
        spot=row['spotRef'];finite=spot is not None and np.isfinite(r)
        if self.asset<4:
            if finite:
                if not s['price_segment_valid']:s['log_price']=float(np.log(spot));s['price_closes']=[]
                else:s['log_price']+=float(r)
                s['price_segment_valid']=True
            else:s['price_segment_valid']=False
        else:
            if spot is not None:s['log_price']=float(np.log(spot))
            s['price_segment_valid']=spot is not None
        closes=s['price_closes'][-20:]
        if s['price_segment_valid'] and len(closes)==20:coefficient[2]=[midrank(np.sort(closes),s['log_price']),1.,.5]
        previous=pd.Timestamp(s['previous_session']);gap=max(0,(date-previous).days-1)
        weekdays=max(0,int(np.busday_count(previous.date(),date.date()))-1)
        calendar=np.r_[np.eye(5)[date.weekday()],np.log1p([gap,weekdays]),1.]
        args=[window[None],aux[None],old[None],state[None],np.array(template['context'])[None],
            np.array([self.asset],np.int64),np.array([template['position'] if position==0 else position],float),
            np.array([template['groups']],np.int64),np.array([template['own_groups']],np.int64),calendar[None],returns[None],coefficient[None],policy[None]]
        for a in args:
            if not np.isfinite(a).all():raise FloatingPointError('nonfinite model input after explicit extensions')
        return args,flags,clock

    def predict_at(self, timestamp):
        """Scheduled output without inventing a production observation or feature row."""
        dt=pd.Timestamp(timestamp)
        return self.predict_step({'usymbol':self.symbol,'trade_date':dt.normalize().isoformat(),
                                  'datetime':dt.isoformat()})

    def _fallback(self,row,flags):
        """Carry the latest forecast, or use the frozen historical RV before first output.
        No synthetic row is inserted into feature histories. max_observed and
        max_consumed distinguish model observations from scheduled requests.
        """
        s=copy.deepcopy(self.state);previous=s['last_predictions'];predictions={}
        if previous:
            predictions=copy.deepcopy(previous)
            origin=s.get('last_forecast_origin_datetime',s['last_datetime'])
            flags.append('previous_forecast_carried')
        else:
            clock=min(self.registry,key=lambda c:self.registry[c]['position'])
            value=float(np.exp(.5*s['histories'][clock][-1][0]))
            if not np.isfinite(value) or value<=0:raise ValueError('no valid historical fallback RV')
            for opt in self.models:
                name='shampoo_e5' if opt=='shampoo_adam_graft' else 'adam_e6'
                predictions[name]=predictions[name+'_half0.5']=value
            origin=s['last_datetime'];flags.append('historical_rv_fallback')
        key=row['datetime'];dt=pd.Timestamp(key)
        observed=s.get('max_observed_datetime',s['last_datetime'])
        result={'symbol':self.symbol,'datetime':key,'clock_slot_id':dt.strftime('%H:%M')+'__scheduled',
                'weight_month':self.month,'weight_version':self.version,'max_consumed_datetime':key,
                'max_observed_datetime':observed,'forecast_origin_datetime':origin,
                'stale_seconds':float((dt-pd.Timestamp(origin)).total_seconds()),
                'prediction_mode':'fallback','flags':sorted(set(flags)),**predictions}
        s.update(last_datetime=key,last_predictions=predictions,max_observed_datetime=observed,
                 last_forecast_origin_datetime=origin)
        return s,result

    def predict_step(self,row):
        """Return one output per chronological timestamp, including numeric fallback."""
        try:
            return self._predict_step(row)
        except (FloatingPointError,OverflowError) as exc:
            normalized,input_flags=validate_available_row(row,self.symbol)
            key=normalized['datetime']
            if pd.Timestamp(key)<=pd.Timestamp(self.state['last_datetime']):raise
            s,result=self._fallback(normalized,input_flags+['numeric_failure_fallback',type(exc).__name__])
            fingerprint=hashlib.sha256(json.dumps({'row':normalized,'input_flags':input_flags},sort_keys=True,allow_nan=False).encode()).hexdigest()
            s['retries'][key]={'fingerprint':fingerprint,'output':copy.deepcopy(result)}
            self.state=s;self.last_args=None
            return result

    def _predict_step(self,row):
        row,input_flags=validate_available_row(row,self.symbol);dt=pd.Timestamp(row['datetime'])
        key=row['datetime'];fingerprint=hashlib.sha256(json.dumps({'row':row,'input_flags':input_flags},sort_keys=True,allow_nan=False).encode()).hexdigest()
        retries=self.state['retries']
        if key in retries:
            prior=retries[key]
            legacy_fingerprint=hashlib.sha256(json.dumps(row,sort_keys=True,allow_nan=False).encode()).hexdigest()
            if prior['fingerprint'] not in (fingerprint,legacy_fingerprint):raise ValueError('same timestamp with different production row')
            return copy.deepcopy(prior['output'])
        if dt<=pd.Timestamp(self.state['last_datetime']):raise ValueError('out-of-order production row')
        # Unknown physical events and incomplete records receive explicit fallback,
        # without creating a pseudo-observation in any causal feature window.
        same_day=dt.normalize()==pd.Timestamp(self.state['date']).normalize()
        observed=self.state['observed_today'] if same_day else []
        minute=dt.strftime('%H:%M')
        occurrence=sum(pd.Timestamp(x['datetime']).strftime('%H:%M')==minute for x in observed)
        clock=minute+'__'+str(occurrence)
        opening='09:29' if self.asset<4 else '09:25'
        canonical=(clock in self.registry or (minute==opening and not observed)) and dt.weekday()<5
        fallback_flags=list(input_flags)
        if not canonical:fallback_flags.append('unregistered_clock_fallback')
        if row['ttime'] is None:fallback_flags.append('missing_ttime_fallback')
        if all(row[k] is None for k in ('ret','ttime','spotRef','rvol1D','rvol1hour')):fallback_flags.append('no_production_observation')
        if not canonical or row['ttime'] is None:
            s,result=self._fallback(row,fallback_flags)
            s['retries'][key]={'fingerprint':fingerprint,'output':copy.deepcopy(result)}
            self.state=s;self.last_args=None
            return result
        # Work on a transaction copy. Failures never partially update histories.
        s=copy.deepcopy(self.state);args,flags,clock=self._features(s,row)
        flags.extend(input_flags)
        tensors=[torch.from_numpy(np.ascontiguousarray(a)) for a in args];predictions={}
        previous=self.state['last_predictions']
        previous_origin=pd.Timestamp(self.state.get('last_forecast_origin_datetime',self.state['last_datetime']))
        gap=(dt-previous_origin).total_seconds()
        linked=dt.normalize()==previous_origin.normalize() and 0<gap<=90
        with torch.no_grad():
            for opt,models in self.models.items():
                value=float(np.mean([m(*tensors)[0].item() for m in models]))
                if not np.isfinite(value) or value<=0:raise FloatingPointError('nonpositive/nonfinite model prediction')
                name='shampoo_e5' if opt=='shampoo_adam_graft' else 'adam_e6'
                # Existing half-life .5 physical-row causal EMA; resets at day/lunch/gaps.
                smoothed=.75*value+.25*previous[name+'_half0.5'] if linked and name+'_half0.5' in previous else value
                predictions[name]=value;predictions[name+'_half0.5']=smoothed
        result={'symbol':self.symbol,'datetime':key,'clock_slot_id':clock,'weight_month':self.month,
            'weight_version':self.version,'max_consumed_datetime':key,'flags':sorted(set(flags)),**predictions}
        result.update(prediction_mode='model',max_observed_datetime=key,forecast_origin_datetime=key,stale_seconds=0.)
        s['last_predictions']=predictions;s['last_datetime']=key
        s['max_observed_datetime']=key;s['last_forecast_origin_datetime']=key
        s['retries'][key]={'fingerprint':fingerprint,'output':copy.deepcopy(result)}
        self.state=s;self.last_args=args
        return result


def load_bundle(path,symbol,month,history_state,**kwargs):
    state=read_state(history_state) if isinstance(history_state,(str,Path)) else history_state
    return Predictor(path,symbol,month,state,**kwargs)
