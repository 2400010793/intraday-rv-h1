"""Read-only, chunked signals.rvol_prod acquisition. Credentials stay in memory."""
import getpass,hashlib,json,os
from pathlib import Path
import pandas as pd
from .schema import SYMBOLS


def download(output,start,end,*,symbols=SYMBOLS,chunk_days=30,evaluation=False):
    import pymysql
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    if start.tz is not None or end.tz is not None or start!=start.normalize() or end!=end.normalize() or start>end:raise ValueError('invalid YYYY-MM-DD range')
    if chunk_days<1 or not symbols or not set(symbols).issubset(SYMBOLS):raise ValueError('invalid chunk/symbol selection')
    root=Path(output)
    if root.exists():raise FileExistsError('download requires a new immutable directory')
    user=os.environ.get('RVOL_PROD_USER') or input('MySQL user: ')
    password=os.environ.get('RVOL_PROD_PASSWORD') or getpass.getpass('MySQL password: ')
    host=os.environ.get('RVOL_PROD_HOST')
    if not host:raise ValueError('set RVOL_PROD_HOST')
    database=os.environ.get('RVOL_PROD_DATABASE','signals')
    if not database.replace('_','').isalnum():raise ValueError('invalid database identifier')
    connection=pymysql.connect(host=host,port=int(os.environ.get('RVOL_PROD_PORT','3306')),user=user,password=password,database=database,charset='utf8mb4',connect_timeout=30,read_timeout=600)
    root.mkdir(parents=True);files=[]
    columns=['usymbol','trade_date','datetime','ret','ttime','spotRef','rvol1D','rvol1hour','rvol15min','rvol3D']
    if evaluation:columns+=['predRVol1D']
    manifest={'schema':'h1-source-v1','table':'rvol_prod','columns':columns,'start':str(start.date()),'end':str(end.date()),'timezone':'Asia/Shanghai naive exchange timestamps','credentials_stored':False,'files':files,'status':'running'}
    try:
        for symbol in symbols:
            low=start;parts=[]
            while low<=end:
                high=min(low+pd.Timedelta(days=chunk_days),end+pd.Timedelta(days=1))
                sql=f"SELECT {','.join(columns)} FROM rvol_prod WHERE usymbol=%s AND trade_date>=%s AND trade_date<%s ORDER BY datetime"
                with connection.cursor() as cursor:
                    cursor.execute(sql,(symbol,str(low.date()),str(high.date())));rows=cursor.fetchall()
                part=pd.DataFrame(rows,columns=columns)
                if not part.empty:
                    part['datetime']=pd.to_datetime(part.datetime);part['trade_date']=pd.to_datetime(part.trade_date)
                    for column in columns[3:]:part[column]=pd.to_numeric(part[column],errors='raise').astype(float)
                    if part.datetime.duplicated().any():raise ValueError(f'duplicate source keys: {symbol}')
                    parts.append(part)
                low=high
            if not parts:raise ValueError(f'no source data: {symbol}')
            frame=pd.concat(parts,ignore_index=True)
            if frame.datetime.duplicated().any():raise ValueError(f'duplicate chunk boundary: {symbol}')
            p=root/f'{symbol}.parquet';frame.to_parquet(p,index=False)
            files.append({'path':p.name,'symbol':symbol,'rows':len(frame),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        manifest['status']='complete'
    except Exception as exc:
        manifest['status']='failed';manifest['error_type']=type(exc).__name__;raise
    finally:
        connection.close();(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest
