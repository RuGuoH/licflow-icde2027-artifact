"""Gold program for L069: LGPL-Lib + finetuned BERT combine SaaS sell."""
import gc

from main_case import *

wiki_text = Work("Wikipedia", "data", "raw", "CC-BY-SA-4.0")
bert_model = Work("BERT", "model", "raw", "Apache-2.0")
lgpl_lib_model = Work("LGPL-Lib", "model", "raw", "LGPL-3.0")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

finetuned_bert = finetune(bert_model, wiki_text)
new_work = combine([finetuned_bert, lgpl_lib_model])

new_work.form = "saas"
analysis_success = par.analysis(new_work, open_policy="sell")
