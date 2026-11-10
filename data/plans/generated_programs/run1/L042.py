from main_case import *
stocksnap_img = Work('StockSnap', 'data', 'raw', 'CC0-1.0') # Image
midjourney_img = Work('Midjourney_gen', 'data', 'raw', 'CC-BY-NC-4.0') # Image
thingverse_3d = Work('Thingverse', 'data', 'raw', 'CC-BY-NC-SA-4.0') # 3D Model
ccmixter_music = Work('ccMixter', 'data', 'raw', 'CC-BY-NC-4.0') # Music
whisper_model = Work('Whisper', 'model', 'raw', 'MIT') # Voice2Text
stable_diffusion_mode = Work('StableDiffusion', 'model', 'raw', 'CreativeML-OpenRAIL-M') # Text2Image
i2vgen_model = Work('I2VGen-XL', 'model', 'raw', 'CC-BY-NC-ND-4.0') # Image2Video
vimeo_video = Work('Vimeo', 'data', 'raw', 'CC-BY-NC-ND-4.0') # Video
works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)
dataset_a = amalgamate([stocksnap_img, midjourney_img, thingverse_3d])
g1 = generate(ccmixter_music, [whisper_model])
g2 = generate(g1, [stable_diffusion_mode])
dataset_b = generate(g2, [i2vgen_model])
dataset_c = amalgamate([dataset_b, vimeo_video])
new_work = combine([dataset_a, dataset_c])
if 'new_work' not in globals() or globals().get('new_work') is None:
    raise RuntimeError('生成的分析代码未定义 new_work：请将复用链的最终结果赋给 new_work')
if not isinstance(globals()['new_work'], Work):
    raise RuntimeError('new_work 必须是 Work 对象，请检查复用方法返回值是否正确赋值')
new_work.form = 'raw'
par.analysis(new_work, open_policy='share')
new_work.summary()
print('\n---------------------------------------------------------------------------------------------\n')
print(new_work)
import json as _mg_report_json
print('\n---------------------------------------------------------------------------------------------\n')
print(_mg_report_json.dumps(new_work.report_payload(), ensure_ascii=False))