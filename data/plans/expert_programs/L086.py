"""Gold program for L086: one gemma-terms-of-use model, train, sell as a service.

The expert file uses another id suffix. Register it as gemma-terms-of-use, the id
used by the modeling runs.
"""
import gc
from pathlib import Path

import yaml

from main_case import *
from license_parser import License

_license_file = Path(__file__).resolve().parents[1] / "licenses" / "gemma_terms_of_use.yaml"
_license_data = yaml.safe_load(_license_file.read_text(encoding="utf-8"))
_license_data["short_id"] = "gemma-terms-of-use"
_license_data.setdefault("redistribute", [])
_template = next(iter(par.licenses_dict.values()))
par.licenses_dict["gemma-terms-of-use"] = License(_license_data, _template.usages_dict)
if "gemma-terms-of-use" not in par.licenses_list:
    par.licenses_list.append("gemma-terms-of-use")

model = Work("模型丁", "model", "raw", "gemma-terms-of-use")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

new_work = train(model)
new_work.form = "saas"
analysis_success = par.analysis(new_work, open_policy="sell")
