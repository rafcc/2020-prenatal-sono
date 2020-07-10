#coding: UTF-8
from ctypes import *
import math
import random
import csv
import os
from lib.pascal_voc_io import PascalVocWriter
from lib import write_csv

cfg_path = "cfg_angledv2_20190803/yolov2-obj.cfg"
weights_path = "backup_angled20190803/yolov2-obj_25000.weights"
meta_data_path = "cfg_angledv2_20190803/obj.data"
darknet_lib_path = "/home/fsakai/darknet-master/libdarknet.so"
folder_name = "20190803_roc/"
normal_image_path = "/share/fujitsu/benchmark/dataset/v_validation/normal"
chd_image_path = "/share/fujitsu/benchmark/dataset/v_validation/abnormal
detection_thresh = 0.01

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

lib = CDLL(darknet_lib_path, RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=detection_thresh, hier_thresh=detection_thresh, nms=.4):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        tmp_res = None
        tmp_score = 0.0
        for i in range(meta.classes):
             if dets[j].prob[i] > tmp_score:
                tmp_score = dets[j].prob[i]
                b = dets[j].bbox
                tmp_res = (int(i),meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h))
        if tmp_res != None:
           res.append(tmp_res)
    res = sorted(res, key=lambda x: -x[2])
    free_image(im)
    free_detections(dets, num)
    return res
def detected_objs(r):
    obj_list = [0 for i in range(21)]
    for index, result in enumerate(r):
        obj_list[result[0]] = result[2]
    return obj_list

if __name__ == "__main__":
    net = load_net(cfg_path, weights_path, 0)
    meta = load_meta(meta_data_path)
    size = (1136,852,3)
    lines = []
    score = []
    dobjs = []
    roc_scores_file = folder_name + "v_roc_scores.csv"
    roc_lines = []
    for nc in rang(2):
        if(nc == 0):
            image_path = normal_image_path + "/n"
            out_file_name = folder_name + "v_tests/n"
        if(nc == 1):
            image_path = chd_image_path + "/chd"
            out_file_name = folder_name + "v_tests/chd"
        for j in range(1,2000):
            filename = out_file_name + str(j) + ".csv"
            lines = []
            score = []
            dobjs = []
            count_v = [0,0,0,0]
            count_v_sum = 0
            for i in range(1000):
                scores = []
                dobjs = []
                image = image_path + str(j) +"/" + str(int(i + 1)).zfill(5) + ".jpg"
                if(not(os.path.isfile(image))):
                    continue
                r = detect(net, meta, image)
                objs = detected_objs(r)
                lines.append(objs)
                if(objs[8] >= 0.01):
                    count_v[0] += 1
                if(objs[9] >= 0.01):
                    count_v[1] += 1
                if(objs[10] >= 0.01):
                    count_v[2] += 1
                if(objs[17] >= 0.01):
                    count_v[3] += 1
            if lines == []:
                continue
            for ic in range(4):
                count_v_sum += count_v[ic]
            v_roc_lines.append([1,1.0 - count_v_sum/80.0,j])
            write_csv(lines,filename)
