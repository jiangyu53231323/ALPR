import json


def get_result_json(json_dir):
    # file = open(json_dir, "rb")
    # fileJson = json.load(file)
    data = {}
    with open(json_dir, 'r') as f:
        dataset = json.load(f)
    assert type(dataset) == dict, 'lpd file format {} not supported'.format(type(dataset))
    for d in dataset["results"]:
        data[d['image_id']] = d
    return data

def get_ccpd_json(json_dir):
    # file = open(json_dir, "rb")
    # fileJson = json.load(file)
    data = {}
    with open(json_dir, 'r') as f:
        dataset = json.load(f)
    assert type(dataset) == dict, 'lpd file format {} not supported'.format(type(dataset))
    for d in dataset["images"]:
        data[d['id']] = d
    return data

results_path = ''
ccpd_path = ''
results = get_result_json(results_path)
ccpd = get_ccpd_json(ccpd_path)

