"""Gold program for L071: BSD-Model + I2VGen-XL combine sell."""
import gc

from main_case import *

bsd_model = Work("BSD-Model", "model", "raw", "BSD-3-Clause")
i2vgen_model = Work("I2VGen-XL", "model", "raw", "CC-BY-NC-ND-4.0")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

new_work = combine([bsd_model, i2vgen_model])

new_work.form = "raw"
analysis_success = par.analysis(new_work, open_policy="sell")
