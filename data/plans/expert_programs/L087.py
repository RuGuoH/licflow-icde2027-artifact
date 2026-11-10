"""Gold program for L087: one 0BSD model, train, sell the weights.

0BSD is not in the built-in license list. Load the expert entry and register it
under the same id used by the modeling runs.
"""
import gc
from pathlib import Path

import yaml

from main_case import *
from license_parser import License

_license_file = Path(__file__).resolve().parents[1] / "licenses" / "0bsd.yaml"
_license_data = yaml.safe_load(_license_file.read_text(encoding="utf-8"))
_license_data["short_id"] = "0BSD"
_license_data.setdefault("redistribute", [])
_template = next(iter(par.licenses_dict.values()))
par.licenses_dict["0BSD"] = License(_license_data, _template.usages_dict)
if "0BSD" not in par.licenses_list:
    par.licenses_list.append("0BSD")

model = Work("模型戊", "model", "raw", "0BSD")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

new_work = train(model)
new_work.form = "raw"
analysis_success = par.analysis(new_work, open_policy="sell")
