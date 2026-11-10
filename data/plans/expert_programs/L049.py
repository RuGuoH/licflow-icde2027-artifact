"""Gold program for L049: L040 variant sell→share."""
import gc

from main_case import *

stocksnap_img = Work("StockSnap", "data", "raw", "CC0-1.0")
midjourney_img = Work("Midjourney_gen", "data", "raw", "CC-BY-NC-4.0")
thingverse_3d = Work("Thingverse", "data", "raw", "CC-BY-NC-SA-4.0")
ccmixter_music = Work("ccMixter", "data", "raw", "CC-BY-NC-4.0")
whisper_model = Work("Whisper", "model", "raw", "MIT")
baize_model = Work("Baize", "model", "raw", "GPL-3.0")
stable_diffusion_mode = Work("StableDiffusion", "model", "raw", "CreativeML-OpenRAIL-M")
i2vgen_model = Work("I2VGen-XL", "model", "raw", "CC-BY-NC-ND-4.0")
vimeo_video = Work("Vimeo", "data", "raw", "CC-BY-NC-ND-4.0")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

dataset_a = amalgamate([stocksnap_img, midjourney_img, thingverse_3d])
g1 = generate(ccmixter_music, whisper_model)
g2 = generate(g1, baize_model)
g3 = generate(g2, stable_diffusion_mode)
dataset_b = generate(g3, i2vgen_model)
dataset_c = amalgamate([dataset_b, vimeo_video])
new_work = combine([dataset_a, dataset_c])

new_work.form = "raw"
analysis_success = par.analysis(new_work, open_policy="share")
