"""Gold program for L061 resolved: Adobe Research License distill amalgamate sell."""
import gc
from pathlib import Path

import yaml

from main_case import *
from license_parser import License

_bench_root = Path(__file__).resolve().parents[1]
_license_file = _bench_root / "licenses" / "adobe_research.yaml"
if _license_file.is_file():
    _license_data = yaml.safe_load(_license_file.read_text(encoding="utf-8"))
    _template = next(iter(par.licenses_dict.values()))
    _sid = _license_data["short_id"]
    par.licenses_dict[_sid] = License(_license_data, _template.usages_dict)
    if _sid not in par.licenses_list:
        par.licenses_list.append(_sid)

bloom_model = Work("BLOOM", "model", "raw", "BigScience-BLOOM-RAIL-1.0")
bert_model = Work("BERT", "model", "raw", "Apache-2.0")
adobe_model = Work("Adobe-Model", "model", "raw", "adobe-research-license_user_license")
bigtranslate_model = Work("BigTranslate", "model", "raw", "GPL-3.0")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

distilled_bert = distill([bloom_model], bert_model)
new_work = amalgamate([distilled_bert, adobe_model, bigtranslate_model])

new_work.form = "raw"
analysis_success = par.analysis(new_work, open_policy="sell")
