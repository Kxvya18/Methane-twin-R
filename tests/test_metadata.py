import json
from methanetwin_r.data.metadata import parse_emit_plume_metadata

def test_metadata_parser_extracts_emission_and_source(tmp_path):
    p=tmp_path/"x.json"
    p.write_text(json.dumps({
        "type":"Feature",
        "geometry":{"type":"Point","coordinates":[-102,33]},
        "properties":{
            "estimated_emission_rate_kg_hr":1234.5,
            "emission_uncertainty_kg_hr":120,
            "source_latitude":33.01,
            "source_longitude":-102.02
        }
    }))
    x=parse_emit_plume_metadata(p)
    assert x["emission_rate_value"]==1234.5
    assert x["source_latitude"]==33.01
    assert x["geometry_centroid_lon"]==-102
