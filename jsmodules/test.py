import json


with open('sample_data.json', 'rb') as f:
    data = json.load(f)

print(type(data['features']))

for image in data['features']:
    for item in image['bands']:
        if item['id'] != "LST":
            continue
        print(item)
