"""Gold program for R001: Mistral Large-2 Research amalgamate sell (legal_real M4)."""

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

mistral7b_model = Work("Mistral-7B", "model", "raw", "Apache-2.0")
mistral_large2_model = Work(
    "Mistral-Large-2", "model", "raw", "mistral-ai-research-license_user_license"
)

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

new_work = amalgamate([mistral7b_model, mistral_large2_model])

new_work.form = "raw"
analysis_success = par.analysis(new_work, open_policy="sell")
