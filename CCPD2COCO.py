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


def init_coco():
    global coco
    # coco = dict()
    coco['images'] = []
    coco['type'] = 'instances'
    coco['annotations'] = []
    category_item['supercategory'] = 'none'
    category_item_id = 1
    category_item['id'] = category_item_id
    category_item['name'] = "License Plate"
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

        image_file = os.path.join(image_path, f)
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

        lp_number = list(map(int, f.split('-')[4].split('.')[0].split('_')))

        if f not in image_set:
            current_image_id = addImgItem(f)
        current_category_id = 1
        addAnnoItem(current_image_id, current_category_id, bbox, vertices_locations)

        print(image_file)
    print('===运行结束===')


if __name__ == '__main__':
    image_path = 'F:\code_download\CCPD2019\ccpd_rotate'
    train_json_file = './ccpd_base_train2020.json'
    test_json_file = './ccpd_rotate_val2020.json'
    # './pascal_trainval0712.json'

    image_path_list = os.listdir(image_path)
    # 随机选取80%的数据作为训练集
    train_number = int(len(image_path_list) * 0.8)
    train_number = 10
    train_list = random.sample(image_path_list, train_number)
    test_list = list(set(image_path_list).difference(set(train_list)))

    # parseImageName(train_list)
    # json.dump(coco, open(train_json_file, 'w'))

    print('init coco')
    init_coco()
    parseImageName(image_path_list)
    json.dump(coco, open(test_json_file, 'w'))
