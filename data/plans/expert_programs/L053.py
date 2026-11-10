"""Gold program for L053 resolved: BigScience-BLOOM-RAIL-1.0 MoE SaaS."""
import gc

from main_case import *

wiki_text = Work("Wikipedia", "data", "raw", "CC-BY-SA-4.0")
bert_model = Work("BERT", "model", "raw", "Apache-2.0")
baize_model = Work("Baize", "model", "raw", "GPL-3.0")
bloom_model = Work("BLOOM", "model", "raw", "BigScience-BLOOM-RAIL-1.0")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

finetuned_bert = finetune(bert_model, wiki_text)
new_work = combine([finetuned_bert, baize_model, bloom_model])

new_work.form = "saas"
analysis_success = par.analysis(new_work, open_policy="sell")
