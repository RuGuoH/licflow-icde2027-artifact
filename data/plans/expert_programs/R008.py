"""Gold program for R008: finetune Codex on MIT, Apache-2.0, and GPL-3.0 code, sell as a service."""
import gc

from main_case import *

mit_code = Work("MIT-Corpus", "software", "raw", "MIT")
apache_code = Work("Apache-Corpus", "software", "raw", "Apache-2.0")
gpl_code = Work("GPL-Corpus", "software", "raw", "GPL-3.0")
codex_model = Work("Codex", "model", "raw", "Apache-2.0")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

code_corpus = amalgamate([mit_code, apache_code, gpl_code])
new_work = finetune(codex_model, code_corpus)

new_work.form = "saas"
analysis_success = par.analysis(new_work, open_policy="sell")
