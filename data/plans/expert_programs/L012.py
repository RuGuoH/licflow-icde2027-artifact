"""Gold program for L012: finetune MaskFormer MoE sell."""
import gc

from main_case import *

flickr_img = Work("Flickr", "data", "raw", "CC-BY-NC-SA-4.0")
maskformer_model = Work("MaskFormer", "model", "raw", "CC-BY-NC-4.0")
i2vgen_model = Work("I2VGen-XL", "model", "raw", "CC-BY-NC-ND-4.0")
detr_model = Work("DETR", "model", "raw", "Apache-2.0")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

finetuned_maskformer = finetune(maskformer_model, flickr_img)
new_work = combine([finetuned_maskformer, i2vgen_model, detr_model])

new_work.form = "raw"
analysis_success = par.analysis(new_work, open_policy="sell")
