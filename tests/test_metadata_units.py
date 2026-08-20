import json
from methanetwin_r.data.metadata import parse_emit_plume_metadata

def test_ambiguous_emission_units_not_silently_called_kg_h(tmp_path):
    p=tmp_path/"x.json"
    p.write_text(json.dumps({"properties":{"emission_rate":12.3}}))
    x=parse_emit_plume_metadata(p)
    assert x["emission_rate_value"] is None
    assert x["emission_rate_raw_value"]==12.3
    assert x["emission_unit_status"]=="ambiguous"
