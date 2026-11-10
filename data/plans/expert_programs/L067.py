"""Gold program for L067: AGPL-Tool + Baize combine SaaS sell."""
import gc

from main_case import *

agpl_tool_model = Work("AGPL-Tool", "model", "raw", "AGPL-3.0")
baize_model = Work("Baize", "model", "raw", "GPL-3.0")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

new_work = combine([agpl_tool_model, baize_model])

new_work.form = "saas"
analysis_success = par.analysis(new_work, open_policy="sell")
