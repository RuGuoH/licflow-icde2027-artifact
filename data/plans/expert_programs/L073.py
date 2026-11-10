"""Gold program for L073: OpenClipart CC0 + Flickr amalgamate sell."""
import gc

from main_case import *

open_clipart_img = Work("OpenClipart", "data", "raw", "CC0-1.0")
flickr_img = Work("Flickr", "data", "raw", "CC-BY-NC-SA-4.0")

works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)

new_work = amalgamate([open_clipart_img, flickr_img])

new_work.form = "raw"
analysis_success = par.analysis(new_work, open_policy="sell")
