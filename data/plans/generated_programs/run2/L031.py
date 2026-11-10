from main_case import *
bloom_model = Work('BLOOM', 'model', 'raw', 'BigScience-BLOOM-RAIL-1.0') # Text Generation
bert_model = Work('BERT', 'model', 'raw', 'Apache-2.0') # Text
llama2_model = Work('Llama2', 'model', 'raw', 'Llama2') # Text Generation
bigtranslate_model = Work('BigTranslate', 'model', 'raw', 'GPL-3.0') # Text Translation
works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)
distilled_bert = distill(bloom_model, bert_model)
new_work = amalgamate([distilled_bert, llama2_model, bigtranslate_model])
if 'new_work' not in globals() or globals().get('new_work') is None:
    raise RuntimeError('生成的分析代码未定义 new_work：请将复用链的最终结果赋给 new_work')
if not isinstance(globals()['new_work'], Work):
    raise RuntimeError('new_work 必须是 Work 对象，请检查复用方法返回值是否正确赋值')
new_work.form = 'raw'
par.analysis(new_work, open_policy='sell')
new_work.summary()
print('\n---------------------------------------------------------------------------------------------\n')
print(new_work)
import json as _mg_report_json
print('\n---------------------------------------------------------------------------------------------\n')
print(_mg_report_json.dumps(new_work.report_payload(), ensure_ascii=False))