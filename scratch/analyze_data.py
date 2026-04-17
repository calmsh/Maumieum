import json

with open('data/m_1315694.json', encoding='utf-8') as f:
    d = json.load(f)

print('=== 최상위 키 ===')
for key in d.keys():
    val = d[key]
    if isinstance(val, dict):
        print('  [{}] (dict) -> 서브키: {}'.format(key, list(val.keys())))
    elif isinstance(val, list):
        sample = str(val[0])[:80] if val else 'empty'
        print('  [{}] (list, 길이={}) -> 샘플: {}'.format(key, len(val), sample))
    else:
        print('  [{}] = {}'.format(key, str(val)[:80]))

print()
print('=== assessment 내부 설문 종류 ===')
for k in d['assessment'].keys():
    dialog = d['assessment'][k].get('dialog', [])
    print('  {} (항목 {}개)'.format(k, len(dialog)))

print()
print('=== hrv_deep 내부 키 ===')
for k in d['hrv_deep'].keys():
    val = d['hrv_deep'][k]
    if isinstance(val, dict):
        sub = list(val.keys())
        print('  [{}] -> 서브키: {}'.format(k, sub))
        for sk in sub:
            sv = val[sk]
            if isinstance(sv, list) and sv:
                print('    [{}] list 길이={}, 샘플 항목 키={}'.format(sk, len(sv), list(sv[0].keys())))
            elif isinstance(sv, dict):
                print('    [{}] dict 키={}'.format(sk, list(sv.keys())))
            else:
                print('    [{}] = {}'.format(sk, str(sv)[:60]))
    else:
        print('  [{}] = {}'.format(k, str(val)[:80]))

print()
print('=== user_info 내부 키 ===')
if 'user_info' in d:
    for k, v in d['user_info'].items():
        print('  {}: {}'.format(k, v))
else:
    print('  (user_info 없음)')

# 상위 레벨에서 추가 키 확인
print()
print('=== 전체 최상위 키 목록 ===')
print(list(d.keys()))
