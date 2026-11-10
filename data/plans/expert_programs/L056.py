"""Gold program for L056 resolved: Llama-3.1 community license distill amalgamate."""
import gc
from pathlib import Path

import yaml

from main_case import *
from license_parser import License

_bench_root = Path(__file__).resolve().parents[1]
_license_file = _bench_root / "licenses" / "llama_3_1_community.yaml"
if _license_file.is_file():
    _license_data = yaml.safe_load(_license_file.read_text(encoding="utf-8"))
    _template = next(iter(par.licenses_dict.values()))
    _sid = _license_data["short_id"]
    par.licenses_dict[_sid] = License(_license_data, _template.usages_dict)
    if _sid not in par.licenses_list:
        par.licenses_list.append(_sid)

bloom_model = Work("BLOOM", "model", "raw", "BigScience-BLOOM-RAIL-1.0")
bert_model = Work("BERT", "model", "raw", "Apache-2.0")
llama31_model = Work("Llama-3.1", "model", "raw", "llama-3.1-community-license-agreement")
bigtranslate_model = Work("BigTranslate", "model", "raw", "GPL-3.0")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

distilled_bert = distill([bloom_model], bert_model)
new_work = amalgamate([distilled_bert, llama31_model, bigtranslate_model])

new_work.form = "raw"
analysis_success = par.analysis(new_work, open_policy="sell")
