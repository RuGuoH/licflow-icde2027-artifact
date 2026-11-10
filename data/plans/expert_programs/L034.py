"""Gold program for L034: L030 variant with open_policy=share."""
import gc

from main_case import *

wikimedia_img = Work("Wikimedia", "data", "raw", "CC-BY-SA-4.0")
i2vgen_model = Work("I2VGen-XL", "model", "raw", "CC-BY-NC-ND-4.0")
xclip_model = Work("X-Clip", "model", "raw", "MIT")
maskformer_model = Work("MaskFormer", "model", "raw", "CC-BY-NC-4.0")
whisper_model = Work("Whisper", "model", "raw", "MIT")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

distilled_x_clip = distill([wikimedia_img, i2vgen_model], xclip_model)
new_work = amalgamate([distilled_x_clip, maskformer_model, whisper_model])

new_work.form = "raw"
analysis_success = par.analysis(new_work, open_policy="share")
