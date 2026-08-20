import pandas as pd
from methanetwin_r.data.splits import assign_group_splits

def test_geographic_group_not_split_across_partitions():
    rows=[]
    for g,(lat,lon) in enumerate([(0,0),(10,10),(20,20),(30,30),(40,40),(50,50)]):
        for j in range(3):
            rows.append({"sample_id":f"{g}_{j}","centroid_lat":lat+.01*j,"centroid_lon":lon+.01*j})
    d=assign_group_splits(pd.DataFrame(rows),bin_deg=5)
    assert d.groupby("region_id")["split"].nunique().max()==1
