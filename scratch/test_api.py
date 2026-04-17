import json
from api import save_user_data, UserData
payload = UserData(
    user_id='guest_user',
    data={'custom_activities': json.dumps([{"name": "게임"}])}
)
res = save_user_data(payload)
print(res)
