"""Gold program for R004: Ghostscript GPL amalgamate sell (legal_real S3 Artifex)."""

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

ghostscript_lib = Work("Ghostscript", "software", "raw", "GPL-3.0")
hancom_office = Work(
    "Hancom-Office", "software", "raw", "adobe-research-license_user_license"
)

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

new_work = amalgamate([ghostscript_lib, hancom_office])

new_work.form = "raw"
analysis_success = par.analysis(new_work, open_policy="sell")
