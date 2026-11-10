"""Gold program for L083: one MIT model, train, sell the weights."""
import gc

from main_case import *

model = Work("模型甲", "model", "raw", "MIT")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

new_work = train(model)
new_work.form = "raw"
analysis_success = par.analysis(new_work, open_policy="sell")
