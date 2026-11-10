from main_case import *
mit = Work('MIT 开源代码', 'software', 'raw', 'MIT')
apache_2_0 = Work('Apache-2.0 开源代码', 'software', 'raw', 'Apache-2.0')
gpl_3_0 = Work('GPL-3.0 开源代码', 'software', 'raw', 'GPL-3.0')
codex_model = Work('Codex', 'model', 'raw', 'Apache-2.0')
works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)
code_corpus = amalgamate([mit, apache_2_0, gpl_3_0])
finetuned_codex = finetune(codex_model, code_corpus)
new_work = finetuned_codex
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