"""Gold program for L002: embed arxiv/stackexchange + combine sell (case 1.2)."""
import gc

from main_case import *

arxiv_text = Work("arXiv", "data", "raw", "CC-BY-NC-SA-4.0")
stack_exchange_text = Work("StackExchange", "data", "raw", "CC-BY-SA-4.0")
nllb_model = Work("NLLB-200", "model", "raw", "Apache-2.0")
deep_sequoia_text = Work("Deep-sequoia", "data", "raw", "LGPL-LR")
free_law_text = Work("FreeLaw", "data", "raw", "CC-BY-ND-4.0")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

translated = [embed(arxiv_text, nllb_model), embed(stack_exchange_text, nllb_model)]
new_work = combine(translated + [deep_sequoia_text, free_law_text])

new_work.form = "raw"
analysis_success = par.analysis(new_work, open_policy="sell")
