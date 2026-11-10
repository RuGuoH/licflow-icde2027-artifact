"""Gold program for L043: hybrid without Midjourney sell."""
import gc

from main_case import *

stocksnap_img = Work("StockSnap", "data", "raw", "CC0-1.0")
thingverse_3d = Work("Thingverse", "data", "raw", "CC-BY-NC-SA-4.0")
ccmixter_music = Work("ccMixter", "data", "raw", "CC-BY-NC-4.0")
whisper_model = Work("Whisper", "model", "raw", "MIT")
stable_diffusion_mode = Work("StableDiffusion", "model", "raw", "CreativeML-OpenRAIL-M")
i2vgen_model = Work("I2VGen-XL", "model", "raw", "CC-BY-NC-ND-4.0")
vimeo_video = Work("Vimeo", "data", "raw", "CC-BY-NC-ND-4.0")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

dataset_a = amalgamate([stocksnap_img, thingverse_3d])
g1 = generate(ccmixter_music, whisper_model)
g2 = generate(g1, stable_diffusion_mode)
final_dataset_b = generate(g2, i2vgen_model)
dataset_c = amalgamate([final_dataset_b, vimeo_video])
new_work = combine([dataset_a, dataset_c])

new_work.form = "raw"
analysis_success = par.analysis(new_work, open_policy="sell")
