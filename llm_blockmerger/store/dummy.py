import json

json_string = '{"a": [1, 2, 3]}'
data = json.loads(json_string)
print(data['a'])