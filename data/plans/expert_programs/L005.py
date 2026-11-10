"""Gold program for L005: L001 variant with open_policy=share."""
import gc

from main_case import *

deep_sequoia_text = Work("Deep-sequoia", "data", "raw", "LGPL-LR")
pubmed_text = Work("PubMed", "data", "raw", "CC-BY-NC-ND-4.0")
nllb_model = Work("NLLB-200", "model", "raw", "Apache-2.0")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

translated = [embed(ds, nllb_model) for ds in [deep_sequoia_text, pubmed_text]]
new_work = combine(translated + [nllb_model])

new_work.form = "raw"
analysis_success = par.analysis(new_work, open_policy="share")
