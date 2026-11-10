"""Gold program for R007: Linksys firmware, GPL-2.0, sell the weights."""
import gc
from pathlib import Path

import yaml

from main_case import *
from license_parser import License

_license_file = Path(__file__).resolve().parents[1] / "licenses" / "gpl_2_0.yaml"
_license_data = yaml.safe_load(_license_file.read_text(encoding="utf-8"))
_license_data["short_id"] = "GPL-2.0"
_template = next(iter(par.licenses_dict.values()))
par.licenses_dict["GPL-2.0"] = License(_license_data, _template.usages_dict)
if "GPL-2.0" not in par.licenses_list:
    par.licenses_list.append("GPL-2.0")

linux_kernel = Work("Linux-kernel", "software", "binary", "GPL-2.0")
busybox = Work("BusyBox", "software", "binary", "GPL-2.0")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

embedded_linux = embed(linux_kernel, busybox)
new_work = combine([embedded_linux, busybox])

analysis_success = par.analysis(new_work, open_policy="sell")
