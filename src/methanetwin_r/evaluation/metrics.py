import numpy as np
from sklearn.metrics import precision_recall_fscore_support,roc_auc_score,average_precision_score,mean_absolute_error,mean_squared_error

def binary_metrics(y,p,threshold=0.5):
    y=np.asarray(y,int); p=np.asarray(p,float); pred=(p>=threshold).astype(int)
    pr,re,f,_=precision_recall_fscore_support(y,pred,average="binary",zero_division=0)
    return {"precision":float(pr),"recall":float(re),"f1":float(f),
            "auroc":float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None,
            "average_precision":float(average_precision_score(y,p)) if len(np.unique(y))>1 else None}

def mask_metrics(true,pred,threshold=0.5):
    t=np.asarray(true)>0.5; p=np.asarray(pred)>=threshold
    inter=(t&p).sum(); union=(t|p).sum()
    iou=inter/union if union else 1.0
    dice=2*inter/(t.sum()+p.sum()) if (t.sum()+p.sum()) else 1.0
    return {"iou":float(iou),"dice":float(dice)}

def regression_metrics(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    return {"mae":float(mean_absolute_error(y,p)),
            "rmse":float(np.sqrt(mean_squared_error(y,p))),
            "mape":float(np.mean(np.abs(y-p)/np.maximum(np.abs(y),1e-6)))}

def haversine_km(lat1,lon1,lat2,lon2):
    R=6371.0088
    a1,a2=np.deg2rad(lat1),np.deg2rad(lat2)
    da=np.deg2rad(lat2-lat1); dl=np.deg2rad(lon2-lon1)
    h=np.sin(da/2)**2+np.cos(a1)*np.cos(a2)*np.sin(dl/2)**2
    return float(2*R*np.arcsin(np.sqrt(h)))
