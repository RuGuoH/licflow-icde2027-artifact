"""Gold program for L004: embed Deep-sequoia/FreeLaw + combine share."""
import gc

from main_case import *

deep_sequoia_text = Work("Deep-sequoia", "data", "raw", "LGPL-LR")
free_law_text = Work("FreeLaw", "data", "raw", "CC-BY-ND-4.0")
nllb_model = Work("NLLB-200", "model", "raw", "Apache-2.0")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

translated = [embed(deep_sequoia_text, nllb_model), embed(free_law_text, nllb_model)]
new_work = combine(translated)

new_work.form = "raw"
analysis_success = par.analysis(new_work, open_policy="share")
