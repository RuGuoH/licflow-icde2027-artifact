"""Gold program for L088: one mg0-2.0 model, train, sell the weights.

The expert file uses another id suffix. Register it as mg0-2.0, the id used by
the modeling runs.
"""
import gc
from pathlib import Path

import yaml

from main_case import *
from license_parser import License

_license_file = Path(__file__).resolve().parents[1] / "licenses" / "mg0_2_0.yaml"
_license_data = yaml.safe_load(_license_file.read_text(encoding="utf-8"))
_license_data["short_id"] = "mg0-2.0"
_license_data.setdefault("redistribute", [])
_template = next(iter(par.licenses_dict.values()))
par.licenses_dict["mg0-2.0"] = License(_license_data, _template.usages_dict)
if "mg0-2.0" not in par.licenses_list:
    par.licenses_list.append("mg0-2.0")

model = Work("模型己", "model", "raw", "mg0-2.0")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

new_work = train(model)
new_work.form = "raw"
analysis_success = par.analysis(new_work, open_policy="sell")
