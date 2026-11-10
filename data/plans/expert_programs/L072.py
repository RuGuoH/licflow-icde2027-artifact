"""Gold program for L072: OpenRAIL++-Model + Baize combine sell."""
import gc

from main_case import *

openrail_pp_model = Work("OpenRAIL++-Model", "model", "raw", "OpenRAIL++")
baize_model = Work("Baize", "model", "raw", "GPL-3.0")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

new_work = combine([openrail_pp_model, baize_model])

new_work.form = "raw"
analysis_success = par.analysis(new_work, open_policy="sell")
