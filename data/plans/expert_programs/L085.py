"""Gold program for L085: one Llama2 model, generate, share the output only."""
import gc

from main_case import *

model = Work("模型丙", "model", "raw", "Llama2")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

new_work = generate(model)
new_work.form = "raw"
analysis_success = par.analysis(new_work, open_policy="share")
