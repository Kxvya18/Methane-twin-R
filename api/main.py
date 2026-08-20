from pathlib import Path
import tempfile
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from methanetwin_r.config import load_config
from methanetwin_r.data.raster import read_raster,resize_array
from methanetwin_r.service.predictor import Predictor

CONFIG="configs/research.yaml"
app=FastAPI(title="MethaneTwin-R API",version="1.0.0")
_predictor=None

def predictor():
    global _predictor
    if _predictor is None:
        cfg=load_config(CONFIG)
        try:_predictor=Predictor(cfg)
        except FileNotFoundError:
            raise HTTPException(503,"Model not trained. Run the training pipeline first.")
    return _predictor

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/predict-geotiff")
async def predict_geotiff(file:UploadFile=File(...),u10:float=Form(...),v10:float=Form(...)):
    cfg=load_config(CONFIG); size=int(cfg["dataset"]["image_size"])
    suffix=Path(file.filename or "input.tif").suffix or ".tif"
    with tempfile.NamedTemporaryFile(suffix=suffix,delete=False) as f:
        f.write(await file.read()); path=f.name
    ras=read_raster(path); img,valid=resize_array(ras["array"],size,fill=0,order=1)
    out=predictor().predict(img,valid,u10,v10)
    out.pop("mask")
    out["bounds"]=list(ras["bounds"])
    return out
