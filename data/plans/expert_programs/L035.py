"""Gold program for L035: distill only, no amalgamate (L031 simplified)."""
import gc

from main_case import *

bloom_model = Work("BLOOM", "model", "raw", "BigScience-BLOOM-RAIL-1.0")
bert_model = Work("BERT", "model", "raw", "Apache-2.0")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

new_work = distill([bloom_model], bert_model)

new_work.form = "raw"
analysis_success = par.analysis(new_work, open_policy="sell")
