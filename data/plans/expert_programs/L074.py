"""Gold program for L074: OpenCorpus CC-BY-4.0 embed + combine sell."""
import gc

from main_case import *

open_corpus_text = Work("OpenCorpus", "data", "raw", "CC-BY-4.0")
nllb_model = Work("NLLB-200", "model", "raw", "Apache-2.0")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

translated = embed(open_corpus_text, nllb_model)
new_work = combine([translated, nllb_model])

new_work.form = "raw"
analysis_success = par.analysis(new_work, open_policy="sell")
