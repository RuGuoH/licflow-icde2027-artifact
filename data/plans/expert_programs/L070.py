"""Gold program for L070: LGPL-Lib + DETR combine raw sell."""
import gc

from main_case import *

lgpl_lib_model = Work("LGPL-Lib", "model", "raw", "LGPL-3.0")
detr_model = Work("DETR", "model", "raw", "Apache-2.0")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

new_work = combine([lgpl_lib_model, detr_model])

new_work.form = "raw"
analysis_success = par.analysis(new_work, open_policy="sell")
