"""Gold program for L060 resolved: Mistral Research License MoE SaaS."""
import gc
from pathlib import Path

import yaml

from main_case import *
from license_parser import License

_bench_root = Path(__file__).resolve().parents[1]
_license_file = _bench_root / "licenses" / "mistral_research.yaml"
if _license_file.is_file():
    _license_data = yaml.safe_load(_license_file.read_text(encoding="utf-8"))
    _template = next(iter(par.licenses_dict.values()))
    _sid = _license_data["short_id"]
    par.licenses_dict[_sid] = License(_license_data, _template.usages_dict)
    if _sid not in par.licenses_list:
        par.licenses_list.append(_sid)

wiki_text = Work("Wikipedia", "data", "raw", "CC-BY-SA-4.0")
bert_model = Work("BERT", "model", "raw", "Apache-2.0")
mistral_model = Work("Mistral", "model", "raw", "mistral-ai-research-license_user_license")
bloom_model = Work("BLOOM", "model", "raw", "BigScience-BLOOM-RAIL-1.0")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

finetuned_bert = finetune(bert_model, wiki_text)
new_work = combine([finetuned_bert, mistral_model, bloom_model])

new_work.form = "saas"
analysis_success = par.analysis(new_work, open_policy="sell")
