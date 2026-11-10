from main_case import *
wikimedia_img = Work('Wikimedia', 'data', 'raw', 'CC-BY-SA-4.0') # Image
i2vgen_model = Work('I2VGen-XL', 'model', 'raw', 'CC-BY-NC-ND-4.0') # Image2Video
xclip_model = Work('X-Clip', 'model', 'raw', 'MIT') # Video2Text
maskformer_model = Work('MaskFormer', 'model', 'raw', 'CC-BY-NC-4.0') # Image Segmentation
whisper_model = Work('Whisper', 'model', 'raw', 'MIT') # Voice2Text
works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)
distilled_xclip = distill(wikimedia_img, xclip_model, [i2vgen_model])
new_work = amalgamate([distilled_xclip, maskformer_model, whisper_model])
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