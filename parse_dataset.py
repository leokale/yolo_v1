# -*- coding:utf-8 -*-
__author__ = 'Leo.Z'

import xml.etree.ElementTree as ET
import os

annotations_root = "H:/dataset/VOC2007/VOCdevkit/VOC2007/Annotations/"
image_root = "H:/dataset/VOC2007/VOCdevkit/VOC2007/JPEGImages"
classes = ['person',
           'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']


def _parse_annotation(root, image_num):
    xml_file_name = os.path.join(root, "{img_num}.xml".format(img_num=image_num))

    ann_str = image_num + '.jpg'

    with open(xml_file_name) as in_file:
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.iter('object'):
            obj_name = obj.find('name').text
            # 如果name不在类别中，则跳过
            if obj_name not in classes:
                continue
            # 获取该类的索引0-19
            clsid = classes.index(obj_name)
            # 找box框
            box = obj.find('bndbox')
            # 分别获得左上角的xy坐标和右下角的xy坐标
            xmin = box.find('xmin').text
            ymin = box.find('ymin').text
            xmax = box.find('xmax').text
            ymax = box.find('ymax').text

            ann_str += " {} {} {} {} {}".format(xmin, ymin, xmax, ymax, clsid)

    return ann_str + '\n'


# 转换所有的xml，并写成txt文件
def convert_annotations(root, outfile):
    txt_outfile = open(outfile, 'w')
    for filename in os.listdir(root):
        img_num = filename.split('.')[0]
        txt_outfile.write(_parse_annotation(root, img_num))

    txt_outfile.close()


# if __name__ == '__main__':
#     convert_annotations(annotations_root, 'image_list.txt')
