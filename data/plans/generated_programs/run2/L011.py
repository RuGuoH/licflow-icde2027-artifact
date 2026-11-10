from main_case import *
flickr_img = Work('Flickr', 'data', 'raw', 'CC-BY-NC-SA-4.0') # Image
maskformer_model = Work('MaskFormer', 'model', 'raw', 'CC-BY-NC-4.0') # Image Segmentation
i2vgen_model = Work('I2VGen-XL', 'model', 'raw', 'CC-BY-NC-ND-4.0') # Image2Video
detr_model = Work('DETR', 'model', 'raw', 'Apache-2.0') # Image Segmentation
works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)
finetuned_maskformer = finetune(maskformer_model, flickr_img)
moe_model = combine([finetuned_maskformer, i2vgen_model, detr_model])
new_work = moe_model
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