"""Gold program for L033: wikimedia multimodal distill + amalgamate sell."""
import gc

from main_case import *

wikimedia_img = Work("Wikimedia", "data", "raw", "CC-BY-SA-4.0")
i2vgen_model = Work("I2VGen-XL", "model", "raw", "CC-BY-NC-ND-4.0")
xclip_model = Work("X-Clip", "model", "raw", "MIT")
maskformer_model = Work("MaskFormer", "model", "raw", "CC-BY-NC-4.0")
whisper_model = Work("Whisper", "model", "raw", "MIT")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

distilled_xclip = distill([i2vgen_model], xclip_model, aux_works=wikimedia_img)
new_work = amalgamate([distilled_xclip, maskformer_model, whisper_model])

new_work.form = "raw"
analysis_success = par.analysis(new_work, open_policy="sell")
