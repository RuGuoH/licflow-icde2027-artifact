"""Gold program for L054 resolved: CreativeML-OpenRAIL-M generate chain."""
import gc

from main_case import *

jamendo_music = Work("Jamendo", "data", "raw", "CC-BY-NC-ND-4.0")
whisper_model = Work("Whisper", "model", "raw", "MIT")
baize_model = Work("Baize", "model", "raw", "GPL-3.0")
stable_diffusion_mode = Work("StableDiffusion", "model", "raw", "CreativeML-OpenRAIL-M")
i2vgen_model = Work("I2VGen-XL", "model", "raw", "CC-BY-NC-ND-4.0")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

g1 = generate(jamendo_music, whisper_model)
g2 = generate(g1, baize_model)
g3 = generate(g2, stable_diffusion_mode)
new_work = generate(g3, i2vgen_model)

new_work.form = "raw"
analysis_success = par.analysis(new_work, open_policy="share")
