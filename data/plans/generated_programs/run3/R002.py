from main_case import *
wiki_text = Work('Wikipedia', 'data', 'raw', 'CC-BY-SA-4.0') # Corpus
bert_model = Work('BERT', 'model', 'raw', 'Apache-2.0') # Text
llama_3_1 = Work('Llama-3.1', 'model', 'saas', 'llama-3.1-community-license-agreement')
bloom_model = Work('BLOOM', 'model', 'raw', 'BigScience-BLOOM-RAIL-1.0') # Text Generation
works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)
finetuned_bert = finetune(bert_model, wiki_text)
new_work = combine([finetuned_bert, llama_3_1, bloom_model])
if 'new_work' not in globals() or globals().get('new_work') is None:
    raise RuntimeError('生成的分析代码未定义 new_work：请将复用链的最终结果赋给 new_work')
if not isinstance(globals()['new_work'], Work):
    raise RuntimeError('new_work 必须是 Work 对象，请检查复用方法返回值是否正确赋值')
new_work.form = 'saas'
par.analysis(new_work, open_policy='sell')
new_work.summary()
print('\n---------------------------------------------------------------------------------------------\n')
print(new_work)
import json as _mg_report_json
print('\n---------------------------------------------------------------------------------------------\n')
print(_mg_report_json.dumps(new_work.report_payload(), ensure_ascii=False))