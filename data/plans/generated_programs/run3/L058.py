from main_case import *
qwen = Work('Qwen', 'model', 'raw', 'qwen-research-license-agreement_user_license')
baize_model = Work('Baize', 'model', 'raw', 'GPL-3.0') # Chatbot
works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)
new_work = combine([qwen, baize_model])
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