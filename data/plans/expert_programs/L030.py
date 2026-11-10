"""Gold program for L030: OPT-175B distill amalgamate sell (v1.1)."""
import gc

from main_case import *

wikimedia_img = Work("Wikimedia", "data", "raw", "CC-BY-SA-4.0")
opt_model = Work("OPT-175B", "model", "raw", "OPT-175B")
xclip_model = Work("X-Clip", "model", "raw", "MIT")
maskformer_model = Work("MaskFormer", "model", "raw", "CC-BY-NC-4.0")
whisper_model = Work("Whisper", "model", "raw", "MIT")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

distilled_x_clip = distill([wikimedia_img, opt_model], xclip_model)
new_work = amalgamate([distilled_x_clip, maskformer_model, whisper_model])

new_work.form = "raw"
analysis_success = par.analysis(new_work, open_policy="sell")
