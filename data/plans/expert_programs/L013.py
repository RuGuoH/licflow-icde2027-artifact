"""Gold program for L013: finetune BERT MoE raw sell (L010 variant)."""
import gc

from main_case import *

wiki_text = Work("Wikipedia", "data", "raw", "CC-BY-SA-4.0")
bert_model = Work("BERT", "model", "raw", "Apache-2.0")
whisper_model = Work("Whisper", "model", "raw", "MIT")
i2vgen_model = Work("I2VGen-XL", "model", "raw", "CC-BY-NC-ND-4.0")
bloom_model = Work("BLOOM", "model", "raw", "BigScience-BLOOM-RAIL-1.0")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

finetuned_bert = finetune(bert_model, wiki_text)
new_work = combine([finetuned_bert, whisper_model, i2vgen_model, bloom_model])

new_work.form = "raw"
analysis_success = par.analysis(new_work, open_policy="sell")
