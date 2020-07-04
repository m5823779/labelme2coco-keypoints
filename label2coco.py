import os
import json
import numpy as np
import glob
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split
np.random.seed(41)

classname_to_id = {"monitor": 1}

class Lableme2CoCo:
    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        # self.img_id = 0
        self.ann_id = 0

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    def to_coco(self, json_path_list):
        self._init_categories()
        for json_path in tqdm(json_path_list):
            obj = self.read_jsonfile(json_path)
            self.images.append(self._image(obj, json_path))
            shapes = obj['shapes']
            for shape in shapes:
                annotation = self._annotation(shape)

            self.img_id += 1
        instance = {}
        instance['info'] = {'description': 'Monitor Dataset', 'version': 1.0, 'year': 2020}
        instance['license'] = ['Acer']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['supercategory'] = k
            category['id'] = v
            category['name'] = k
            category['keypoint'] = ['left_top', 'right_top', 'right_bottom', 'left_bottom']
            self.categories.append(category)

    def _image(self, obj, path):
        image = {}
        from labelme import utils
        img_x = utils.img_b64_to_arr(obj['imageData'])
        h, w = img_x.shape[:-1]
        image['height'] = h
        image['width'] = w
        self.img_id = int(os.path.basename(path).split(".json")[0])
        image['id'] = self.img_id
        # image['id'] = self.img_id
        image['file_name'] = os.path.basename(path).replace(".json", ".jpg")
        return image

    # 构建COCO的annotation字段
    def _annotation(self, shape):
        label = shape['label']
        points = shape['points']
        if len(points) == 2:
            self.annotation = {}
            self.keypoints = []
            self.num_keypoints = 0
            self.annotations.append(self.annotation)
            self.ann_id += 1
            self.annotation['id'] = self.ann_id
            self.annotation['image_id'] = self.img_id
            self.annotation['category_id'] = int(classname_to_id[label])
            self.annotation['iscrowd'] = 0
            self.annotation['area'] = 1.0
            self.annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
            self.annotation['bbox'] = self._get_box(points)
        elif len(points) == 1:
            self.annotation['keypoints'] = self._get_keypoints(points[0])
        self.annotation['num_keypoints'] = self.num_keypoints

        return self.annotation

    # 读取json文件，返回一个json对象
    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]

    def _get_keypoints(self, points):
        if points[0] == 0 and points[1] == 0:
            visable = 0
        else:
            visable = 2
            self.num_keypoints += 1
        self.keypoints.extend([points[0], points[1], visable])
        return self.keypoints


if __name__ == '__main__':
    labelme_path = "./test/"
    saved_coco_path = "./test/"
    if not os.path.exists("%scoco/annotations/"%saved_coco_path):
        os.makedirs("%scoco/annotations/"%saved_coco_path)
    if not os.path.exists("%scoco/train/"%saved_coco_path):
        os.makedirs("%scoco/train"%saved_coco_path)
    if not os.path.exists("%scoco/val/"%saved_coco_path):
        os.makedirs("%scoco/val"%saved_coco_path)
    json_list_path = glob.glob(labelme_path + "/*.json")
    train_path, val_path = train_test_split(json_list_path, test_size=0.12)
    print("train_n:", len(train_path), 'val_n:', len(val_path))

    l2c_train = Lableme2CoCo()
    train_instance = l2c_train.to_coco(train_path)
    l2c_train.save_coco_json(train_instance, '%scoco/annotations/keypoints_train.json'%saved_coco_path)
    for file in train_path:
        shutil.copy(file.replace("json","jpg"),"%scoco/train/"%saved_coco_path)
    for file in val_path:
        shutil.copy(file.replace("json","jpg"),"%scoco/val/"%saved_coco_path)

    l2c_val = Lableme2CoCo()
    val_instance = l2c_val.to_coco(val_path)
    l2c_val.save_coco_json(val_instance, '%scoco/annotations/keypoints_val.json'%saved_coco_path)