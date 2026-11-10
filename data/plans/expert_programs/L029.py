"""Gold program for L029: 2-step generate chain share."""
import gc

from main_case import *

jamendo_music = Work("Jamendo", "data", "raw", "CC-BY-NC-ND-4.0")
whisper_model = Work("Whisper", "model", "raw", "MIT")
stable_diffusion_mode = Work("StableDiffusion", "model", "raw", "CreativeML-OpenRAIL-M")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

g1 = generate(jamendo_music, whisper_model)
new_work = generate(g1, stable_diffusion_mode)

new_work.form = "raw"
analysis_success = par.analysis(new_work, open_policy="share")
