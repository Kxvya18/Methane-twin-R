import pandas as pd
import numpy as np

def beta_persistence(detections, opportunities, alpha0=1.0, beta0=1.0):
    detections=int(detections); opportunities=int(opportunities)
    failures=max(0,opportunities-detections)
    a=alpha0+detections; b=beta0+failures
    mean=a/(a+b)
    # Normal approximation is adequate for display; report parameters for exact downstream use.
    var=a*b/((a+b)**2*(a+b+1))
    return {"alpha":a,"beta":b,"posterior_mean":mean,
            "approx_95_low":max(0,mean-1.96*np.sqrt(var)),
            "approx_95_high":min(1,mean+1.96*np.sqrt(var))}
