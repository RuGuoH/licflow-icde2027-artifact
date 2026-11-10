"""Gold program for L055 resolved: Llama2 distill amalgamate sell."""
import gc

from main_case import *

bloom_model = Work("BLOOM", "model", "raw", "BigScience-BLOOM-RAIL-1.0")
bert_model = Work("BERT", "model", "raw", "Apache-2.0")
llama2_model = Work("Llama2", "model", "raw", "Llama2")
bigtranslate_model = Work("BigTranslate", "model", "raw", "GPL-3.0")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

distilled_bert = distill([bloom_model], bert_model)
new_work = amalgamate([distilled_bert, llama2_model, bigtranslate_model])

new_work.form = "raw"
analysis_success = par.analysis(new_work, open_policy="sell")
