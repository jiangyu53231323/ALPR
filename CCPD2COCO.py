import xml.etree.ElementTree as ET
import os
import json
import random

coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []

category_item = dict()
category_item['supercategory'] = 'none'
category_item_id = 1
category_item['id'] = category_item_id
category_item['name'] = "License Plate"
coco['categories'].append(category_item)

category_set = dict()
image_set = set()

image_size = {'width': 720, 'height': 1160}

category_item_id = 0
image_id = 20200000000
annotation_id = 0


def init_image_set():
    global image_set
    image_set = set()


def init_coco():
    global coco, image_id
    image_id = 20200000000
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instances'
    coco['annotations'] = []
    category_item['supercategory'] = 'none'
    category_item_id = 1
    category_item['id'] = category_item_id
    category_item['name'] = "License Plate"
    coco['categories'] = []
    coco['categories'].append(category_item)


def addCatItem(name):
    global category_item_id
    category_item = dict()
    category_item['supercategory'] = 'none'
    category_item_id += 1
    category_item['id'] = category_item_id
    category_item['name'] = name
    coco['categories'].append(category_item)
    category_set[name] = category_item_id
    return category_item_id


def addImgItem(file_name):
    global image_id
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    image_id += 1
    image_item = dict()
    image_item['id'] = image_id
    image_item['file_name'] = file_name
    image_item['width'] = image_size['width']
    image_item['height'] = image_size['height']
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id


def addAnnoItem(image_id, category_id, bbox, segmentation):
    global annotation_id
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    # bbox[] is x,y,w,h
    # left_top
    seg.append(segmentation[2][0])
    seg.append(segmentation[2][1])
    # left_bottom
    seg.append(segmentation[1][0])
    seg.append(segmentation[1][1])
    # right_bottom
    seg.append(segmentation[0][0])
    seg.append(segmentation[0][1])
    # right_top
    seg.append(segmentation[3][0])
    seg.append(segmentation[3][1])
    if any((abs(segmentation[2][0] - segmentation[3][0]) < 10, abs(segmentation[1][0] - segmentation[0][0]) < 10,
            abs(segmentation[2][1] - segmentation[1][1]) < 10, abs(segmentation[0][1] - segmentation[3][1]) < 10)):
        print(image_id)

    annotation_item['segmentation'].append(seg)
    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_id += 1
    annotation_item['id'] = annotation_id
    coco['annotations'].append(annotation_item)


def parseImageName(image_path_list):
    n = 0
    for f in image_path_list:
        if not f.endswith('.jpg'):
            continue

        real_file_name = f

        bndbox = dict()
        size = dict()
        current_image_id = None
        current_category_id = None
        file_name = None
        size['width'] = None
        size['height'] = None
        size['depth'] = None

        area = None
        tilt_degree = []
        bbox_coordinates = []
        vertices_locations = []
        lp_number = []

        if f not in image_set:
            current_image_id = addImgItem(f)
        n += 1
        # image_file = os.path.join(image_path, f)
        try:
            area = int(f.split('-')[0])
            tilt_degree = f.split('-')[1].split('_')
            tilt_degree = list(map(int, tilt_degree))
            bbox_coordinates = f.split('-')[2].split('_')
            # bbox_coordinates[[x1,y1],[x2,y2]] 左上角和右下角坐标
            bbox_coordinates = [x.split('&') for x in bbox_coordinates]
            bbox = []
            # bbox[x1,y1,w,h]
            bbox.append(int(bbox_coordinates[0][0]))
            bbox.append(int(bbox_coordinates[0][1]))
            bbox.append(int(bbox_coordinates[1][0]) - int(bbox_coordinates[0][0]))
            bbox.append(int(bbox_coordinates[1][1]) - int(bbox_coordinates[0][1]))

            vertices_locations = f.split('-')[3].split('_')
            vertices_locations = [list(map(int, x.split('&'))) for x in vertices_locations]
            segmentation = []
            current_category_id = 1
            lp_number = list(map(int, f.split('-')[4].split('.')[0].split('_')))
            addAnnoItem(current_image_id, current_category_id, bbox, vertices_locations)
        except:
            current_category_id = 0
            bbox = [0, 0, 0, 0]
            vertices_locations = [[0, 0], [0, 0], [0, 0], [0, 0]]
            addAnnoItem(current_image_id, current_category_id, bbox, vertices_locations)
        if n % 10000 == 0:
            print(n)
        # print(image_file)
    print('===运行结束===')


if __name__ == '__main__':
    image_path_base = 'F:\code_download\CCPD2019\ccpd_base'
    image_path_blur = 'F:\code_download\CCPD2019\ccpd_blur'
    image_path_challenge = 'F:\code_download\CCPD2019\ccpd_challenge'
    image_path_db = 'F:\code_download\CCPD2019\ccpd_db'
    image_path_fn = 'F:\code_download\CCPD2019\ccpd_fn'
    # image_path_np = 'F:\code_download\CCPD2019\ccpd_np'
    image_path_rotate = 'F:\code_download\CCPD2019\ccpd_rotate'
    image_path_tilt = 'F:\code_download\CCPD2019\ccpd_tilt'
    image_path_weather = 'F:\code_download\CCPD2019\ccpd_weather'

    train_json_file = './ccpd_train2020.json'
    val_json_file = './ccpd_val2020.json'
    val_base_json_file = './ccpd_base_val2020.json'
    val_blur_json_file = './ccpd_blur_val2020.json'
    val_challenge_json_file = './ccpd_challenge_val2020.json'
    val_db_json_file = './ccpd_db_val2020.json'
    val_fn_json_file = './ccpd_fn_val2020.json'
    val_rotate_json_file = './ccpd_rotate_val2020.json'
    val_tilt_json_file = './ccpd_tilt_val2020.json'
    val_weather_json_file = './ccpd_weather_val2020.json'
    # './pascal_trainval0712.json'

    image_path_list_base = os.listdir(image_path_base)
    image_path_list_blur = os.listdir(image_path_blur)
    image_path_list_challenge = os.listdir(image_path_challenge)
    image_path_list_db = os.listdir(image_path_db)
    image_path_list_fn = os.listdir(image_path_fn)
    # image_path_list_np = os.listdir(image_path_np)
    image_path_list_rotate = os.listdir(image_path_rotate)
    image_path_list_tilt = os.listdir(image_path_tilt)
    image_path_list_weather = os.listdir(image_path_weather)
    # 随机选取80%的数据作为训练集
    train_number_base = int(len(image_path_list_base) * 0.8)
    train_number_blur = int(len(image_path_list_blur) * 0.8)
    train_number_challenge = int(len(image_path_list_challenge) * 0.8)
    train_number_db = int(len(image_path_list_db) * 0.8)
    train_number_fn = int(len(image_path_list_fn) * 0.8)
    # train_number_np = int(len(image_path_list_np) * 0.8)
    train_number_rotate = int(len(image_path_list_rotate) * 0.8)
    train_number_tilt = int(len(image_path_list_tilt) * 0.8)
    train_number_weather = int(len(image_path_list_weather) * 0.8)

    train_list_base = random.sample(image_path_list_base, train_number_base)
    train_list_blur = random.sample(image_path_list_blur, train_number_blur)
    train_list_challenge = random.sample(image_path_list_challenge, train_number_challenge)
    train_list_db = random.sample(image_path_list_db, train_number_db)
    train_list_fn = random.sample(image_path_list_fn, train_number_fn)
    # train_list_np = random.sample(image_path_list_np, train_number_np)
    train_list_rotate = random.sample(image_path_list_rotate, train_number_rotate)
    train_list_tilt = random.sample(image_path_list_tilt, train_number_tilt)
    train_list_weather = random.sample(image_path_list_weather, train_number_weather)

    test_list_base = list(set(image_path_list_base).difference(set(train_list_base)))
    test_list_blur = list(set(image_path_list_blur).difference(set(train_list_blur)))
    test_list_challenge = list(set(image_path_list_challenge).difference(set(train_list_challenge)))
    test_list_db = list(set(image_path_list_db).difference(set(train_list_db)))
    test_list_fn = list(set(image_path_list_fn).difference(set(train_list_fn)))
    # test_list_np = list(set(image_path_list_np).difference(set(train_list_np)))
    test_list_rotate = list(set(image_path_list_rotate).difference(set(train_list_rotate)))
    test_list_tilt = list(set(image_path_list_tilt).difference(set(train_list_tilt)))
    test_list_weather = list(set(image_path_list_weather).difference(set(train_list_weather)))

    train_list = train_list_base + train_list_blur + train_list_challenge + train_list_db + train_list_fn + train_list_rotate + train_list_tilt + train_list_weather
    test_list = test_list_base + test_list_blur + test_list_challenge + test_list_db + test_list_fn + test_list_rotate + test_list_tilt + test_list_weather

    parseImageName(train_list)
    # json.dump(coco, open(train_json_file, 'w'))

    print('init coco: val')
    init_coco()
    parseImageName(test_list)
    # json.dump(coco, open(val_json_file, 'w'))
    print('init coco:base val')
    init_image_set()
    init_coco()
    parseImageName(test_list_base)
    # json.dump(coco, open(val_base_json_file, 'w'))
    print('init coco:blur val')
    init_image_set()
    init_coco()
    parseImageName(test_list_blur)
    # json.dump(coco, open(val_blur_json_file, 'w'))
    print('init coco:challenge val')
    init_image_set()
    init_coco()
    parseImageName(test_list_challenge)
    # json.dump(coco, open(val_challenge_json_file, 'w'))
    print('init coco:db val')
    init_image_set()
    init_coco()
    parseImageName(test_list_db)
    # json.dump(coco, open(val_db_json_file, 'w'))
    print('init coco:fn val')
    init_image_set()
    init_coco()
    parseImageName(test_list_fn)
    # json.dump(coco, open(val_fn_json_file, 'w'))
    print('init coco:rotate val')
    init_image_set()
    init_coco()
    parseImageName(test_list_rotate)
    # json.dump(coco, open(val_rotate_json_file, 'w'))
    print('init coco:tilt val')
    init_image_set()
    init_coco()
    parseImageName(test_list_tilt)
    # json.dump(coco, open(val_tilt_json_file, 'w'))
    print('init coco:weather val')
    init_image_set()
    init_coco()
    parseImageName(test_list_weather)
    # json.dump(coco, open(val_weather_json_file, 'w'))
