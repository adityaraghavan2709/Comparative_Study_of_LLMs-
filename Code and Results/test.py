import json
json_file = 'tracking_shuffled_objects_seven_objects.json'
def load_examples(json_path: str):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['examples']

data=load_examples(json_file)
print(len(data))