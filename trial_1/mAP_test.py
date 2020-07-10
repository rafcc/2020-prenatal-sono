import xml.etree.ElementTree as ET
import numpy as np
from chainercv.evaluations import eval_detection_voc
import os

g_dir_path = "/home/fsakai/darknet-master/data20190729/validation_xml/n"
p_dir_path = "/home/fsakai/darknet-master/20190804_roc/p_val_xml/n"

classes = ['1_crux',
'2_ventricular_septum',
'3_right_atrium',
'4_tricuspid_valve',
'5_right_ventricle',
'6_left_atrium',
'7_mitral_valve',
'8_left_ventricle',
'9_pulmonary_artery',
'10_ascending_aorta',
'11_superior_vena_cava',
'12_descending_aorta',
'13_stomach',
'14_spine',
'15_umbilical_vein',
'16_inferior_vena_cava',
'17_pulmonary_vein',
'18_ductus_arteriosus',
'19_angled_ventricular_septum',
'20_angled_right_ventricle',
'21_angled_left_ventricle',]

def read_annotation(infile_name,gp):
    in_file = open(infile_name)
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    bbs = []
    scs = []
    clses = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        if(cls_id == 18):
            cls_id = 1
        if(cls_id == 19):
            cls_id = 4
        if(cls_id == 20):
            cls_id = 7
        clses.append(cls_id)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('ymin').text), float(xmlbox.find('xmin').text), float(xmlbox.find('ymax').text), float(xmlbox.find('xmax').text))
        bbs.append(b)
        if(gp == 'p'):
            probability = float(obj.find('probability').text)
            scs.append(probability)

    return bbs,clses,scs

g_bboxes = []
g_labels = []
p_bboxes = []
p_labels = []
p_scores = []

for n in range(1300):
    for i in range(800):
        g_path = g_dir_path + str(n) + "_" + str(i).zfill(5) + ".xml"
        p_path = p_dir_path + str(n) + "_" + str(i).zfill(5) + ".xml"

        if(not os.path.isfile(g_path)):
            continue
        g_bbs,g_clses,_ = read_annotation(g_path,'g')
        p_bbs,p_clses,p_scs = read_annotation(p_path,'p')
        g_bboxes.append(np.array(g_bbs,dtype=np.float32))
        g_labels.append(np.array(g_clses,dtype=np.int32))
        p_bboxes.append(np.array(p_bbs,dtype=np.float32))
        p_labels.append(np.array(p_clses,dtype=np.int32))
        p_scores.append(np.array(p_scs,dtype=np.float32))

bboxes = []
labels = []
scores = []
gt_bboxes = []
gt_labels = []

score = eval_detection_voc(p_bboxes, p_labels, p_scores, g_bboxes, g_labels,iou_thresh=iou_thresh_value)
print(score)
