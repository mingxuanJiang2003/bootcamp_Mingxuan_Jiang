import argparse
import hashlib
import json
import logging
import os
import sys
import time
import pandas as pd

def ensure_dir(p):
    d=os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d,exist_ok=True)

def file_hash(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(8192),b""):
            h.update(chunk)
    return h.hexdigest()

def hash_path(out_path):
    d=os.path.dirname(out_path)
    stem=os.path.splitext(os.path.basename(out_path))[0]
    return os.path.join("checkpoints",f"{stem}.hash")

def should_skip(out_path,input_paths):
    hp=hash_path(out_path)
    if not os.path.exists(out_path) or not os.path.exists(hp):
        return False
    try:
        with open(hp,"r") as f:
            record=json.load(f)
        for p in input_paths:
            if record.get(p,"")!=file_hash(p):
                return False
        return True
    except Exception:
        return False

def write_hash(out_path,input_paths):
    hp=hash_path(out_path)
    ensure_dir(hp)
    record={p:file_hash(p) for p in input_paths}
    with open(hp,"w") as f:
        json.dump(record,f)

def setup_logger(name,log_path):
    ensure_dir(log_path)
    logger=logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh=logging.FileHandler(log_path)
    fmt=logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    if logger.handlers:
        logger.handlers.clear()
    logger.addHandler(fh)
    return logger

def task_ingest(input_path,out_path,logger):
    t=time.time()
    df=pd.read_csv(input_path)
    ensure_dir(out_path)
    df.to_parquet(out_path,index=False)
    write_hash(out_path,[input_path])
    logger.info(f"ingest {input_path} -> {out_path} {len(df)} rows {time.time()-t:.3f}s")

def task_clean(input_path,out_path,logger):
    t=time.time()
    df=pd.read_parquet(input_path)
    df=df.dropna()
    if "date" in df.columns:
        df=df.sort_values("date")
    ensure_dir(out_path)
    df.to_csv(out_path,index=False)
    write_hash(out_path,[input_path])
    logger.info(f"clean {input_path} -> {out_path} {len(df)} rows {time.time()-t:.3f}s")

def task_feature(input_path,out_path,logger):
    t=time.time()
    df=pd.read_csv(input_path)
    if "close" in df.columns:
        df["ret"]=df["close"].pct_change()
        df["vol20"]=df["ret"].rolling(20).std()
    ensure_dir(out_path)
    df.to_csv(out_path,index=False)
    write_hash(out_path,[input_path])
    logger.info(f"feature {input_path} -> {out_path} {len(df)} rows {time.time()-t:.3f}s")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("step",choices=["ingest","clean","feature"])
    p.add_argument("--input",required=True)
    p.add_argument("--out",required=True)
    args=p.parse_args()
    log_path=os.path.join("refactor_demo","logs",f"{args.step}.log")
    logger=setup_logger(args.step,log_path)
    if not os.path.exists(args.input):
        logger.error("missing input")
        sys.exit(2)
    if should_skip(args.out,[args.input]):
        logger.info("skip")
        sys.exit(0)
    if args.step=="ingest":
        task_ingest(args.input,args.out,logger)
    elif args.step=="clean":
        task_clean(args.input,args.out,logger)
    elif args.step=="feature":
        task_feature(args.input,args.out,logger)

if __name__=="__main__":
    main()
