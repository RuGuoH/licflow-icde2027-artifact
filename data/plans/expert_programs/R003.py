"""Gold program for R003: Stability SD3 Community License generate sell (legal_real M5)."""

import gc

from pathlib import Path

import yaml

from main_case import *
from license_parser import License

_bench_root = Path(__file__).resolve().parents[1]
_license_file = _bench_root / "licenses" / "stability_community.yaml"
if _license_file.is_file():
    _license_data = yaml.safe_load(_license_file.read_text(encoding="utf-8"))
    _template = next(iter(par.licenses_dict.values()))
    _sid = _license_data["short_id"]
    par.licenses_dict[_sid] = License(_license_data, _template.usages_dict)
    if _sid not in par.licenses_list:
        par.licenses_list.append(_sid)

flickr_img = Work("Flickr", "data", "raw", "CC-BY-NC-SA-4.0")
sd3_model = Work(
    "Stable-Diffusion-3", "model", "raw", "stability-ai-community-license_user_license"
)

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

new_work = generate(flickr_img, sd3_model)

new_work.form = "raw"
analysis_success = par.analysis(new_work, open_policy="sell")
