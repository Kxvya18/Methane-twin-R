from pathlib import Path
import numpy as np,pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from methanetwin_r.config import load_config
from methanetwin_r.physics.plume import gaussian_plume_2d
from methanetwin_r.service.predictor import Predictor
from methanetwin_r.utils import read_json

CFG_PATH=st.sidebar.text_input("Config","configs/research.yaml")
cfg=load_config(CFG_PATH)

st.set_page_config(page_title="MethaneTwin-R Investigator",layout="wide")
st.title("MethaneTwin-R")
st.caption("Physics-informed methane super-emitter investigation simulator")

@st.cache_resource
def get_predictor(config_path):
    try:return Predictor(load_config(config_path))
    except Exception:return None

pred=get_predictor(CFG_PATH)
mode=st.sidebar.radio("Mode",["Physics Lab","Real EMIT Investigator"])

if mode=="Physics Lab":
    st.subheader("Controlled inverse-problem laboratory")
    c1,c2,c3=st.columns(3)
    q=c1.slider("True emission rate (kg CH₄/h)",50,8000,1500,50)
    wind=c2.slider("Wind speed (m/s)",1.0,15.0,6.0,0.25)
    direction=c3.slider("Wind travel direction (°)",0,359,25)
    c1,c2,c3=st.columns(3)
    sx=c1.slider("Source x",0.10,0.60,0.25,0.01)
    sy=c2.slider("Source y",0.10,0.90,0.55,0.01)
    noise=c3.slider("Retrieval noise",0.0,0.25,0.06,0.01)
    cloud=st.slider("Cloud / invalid fraction",0.0,0.35,0.05,0.01)
    seed=st.number_input("Scenario seed",value=42,step=1)

    if st.button("GENERATE + ANALYZE",type="primary"):
        d=gaussian_plume_2d(size=int(cfg["dataset"]["image_size"]),
                            pixel_size_m=cfg["physics"]["pixel_size_m"],
                            source_xy=(sx,sy),q_kg_h=q,wind_speed=wind,wind_dir_deg=direction,
                            noise=noise,cloud_fraction=cloud,seed=int(seed))
        col1,col2=st.columns(2)
        fig,ax=plt.subplots()
        im=ax.imshow(d["enhancement"]);ax.scatter([sx*(d["enhancement"].shape[1]-1)],[sy*(d["enhancement"].shape[0]-1)],marker="x")
        ax.set_title("Generated methane enhancement");fig.colorbar(im,ax=ax);col1.pyplot(fig)
        if pred is None:
            col2.warning("Train/fine-tune the model before inference. The simulator is generating real new scenarios correctly.")
        else:
            out=pred.predict(d["enhancement"],d["valid"],d["u10"],d["v10"])
            err=np.hypot(out["source_xy"][0]-sx,out["source_xy"][1]-sy)*cfg["dataset"]["image_size"]*cfg["physics"]["pixel_size_m"]
            col2.metric("Detection probability",f'{100*out["detection_probability"]:.1f}%')
            col2.metric("Estimated Q",f'{out["emission_rate_kg_h"]:.0f} kg/h')
            col2.metric("Source error",f'{err:.0f} m')
            if out["emission_interval_kg_h"]:
                col2.write("95% conformal interval:",[round(x,1) for x in out["emission_interval_kg_h"]])
            fig2,ax2=plt.subplots();ax2.imshow(out["mask"]);ax2.scatter([out["source_xy"][0]*(d["enhancement"].shape[1]-1)],
                    [out["source_xy"][1]*(d["enhancement"].shape[0]-1)],marker="x");ax2.set_title("Predicted mask + source");col2.pyplot(fig2)

else:
    st.subheader("Held-out real EMIT scene investigator")
    mp=Path(cfg["paths"]["processed"])/"manifest.parquet"
    if not mp.exists():
        st.warning("Build the real dataset first.")
        st.stop()
    m=pd.read_parquet(mp)
    test=m[(m.split=="test")&(m.detection==1)].reset_index(drop=True)
    if len(test)==0:
        st.warning("No positive held-out test samples are available.")
        st.stop()
    sid=st.selectbox("Scene",test.sample_id.tolist())
    r=test[test.sample_id==sid].iloc[0]
    z=np.load(r.npz_path,allow_pickle=True)
    fig,ax=plt.subplots(); im=ax.imshow(z["enhancement"]);ax.set_title(sid);fig.colorbar(im,ax=ax);st.pyplot(fig)
    st.write("Label origin:",r.label_origin)
    st.write("Source label quality:",r.source_label_quality)
    st.write("Reference emission rate:",None if pd.isna(r.emission_rate_kg_h) else float(r.emission_rate_kg_h))
    if pred is not None:
        u=float(z["u10"]) if np.isfinite(z["u10"]) else 0
        v=float(z["v10"]) if np.isfinite(z["v10"]) else 0
        out=pred.predict(z["enhancement"],z["valid"],u,v)
        c1,c2,c3=st.columns(3)
        c1.metric("Detection",f'{100*out["detection_probability"]:.1f}%')
        c2.metric("Estimated Q",f'{out["emission_rate_kg_h"]:.0f} kg/h')
        c3.metric("Wind",f'{np.hypot(u,v):.1f} m/s')
        fig2,ax2=plt.subplots();ax2.imshow(out["mask"]);ax2.scatter([out["source_xy"][0]*(z["enhancement"].shape[1]-1)],
                [out["source_xy"][1]*(z["enhancement"].shape[0]-1)],marker="x");ax2.set_title("Prediction");st.pyplot(fig2)
