"""Gold program for L068: finetune AGPL-Tool + Whisper MoE SaaS sell."""
import gc

from main_case import *

wiki_text = Work("Wikipedia", "data", "raw", "CC-BY-SA-4.0")
agpl_tool_model = Work("AGPL-Tool", "model", "raw", "AGPL-3.0")
whisper_model = Work("Whisper", "model", "raw", "MIT")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

finetuned_agpl = finetune(agpl_tool_model, wiki_text)
new_work = combine([finetuned_agpl, whisper_model])

new_work.form = "saas"
analysis_success = par.analysis(new_work, open_policy="sell")
