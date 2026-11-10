"""Gold program for L006: triple corpus embed + combine sell."""
import gc

from main_case import *

arxiv_text = Work("arXiv", "data", "raw", "CC-BY-NC-SA-4.0")
stack_exchange_text = Work("StackExchange", "data", "raw", "CC-BY-SA-4.0")
pubmed_text = Work("PubMed", "data", "raw", "CC-BY-NC-ND-4.0")
nllb_model = Work("NLLB-200", "model", "raw", "Apache-2.0")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

translated = [
    embed(arxiv_text, nllb_model),
    embed(stack_exchange_text, nllb_model),
    embed(pubmed_text, nllb_model),
]
new_work = combine(translated)

new_work.form = "raw"
analysis_success = par.analysis(new_work, open_policy="sell")
