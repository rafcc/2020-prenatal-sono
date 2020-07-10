#coding: UTF-8
from ctypes import *
import math
import random
import csv
import os
from lib.pascal_voc_io import PascalVocWriter

cfg_path = "cfg_angledv2_20190803/yolov2-obj.cfg"
weights_path = "backup_angled20190803/yolov2-obj_25000.weights"
meta_data_path = "cfg_angledv2_20190803/obj.data"
label_path = "/home/fsakai/darknet-master/20190804_roc/p_val_xml"
xml_dir_path =  "/home/fsakai/darknet-master/data20190729/validation_xml/n"
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

lib = CDLL("/home/fsakai/darknet-master/libdarknet.so", RTLD_GLOBAL)
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
    print(im.w,im.h)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((int(i),meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[2])
    free_image(im)
    free_detections(dets, num)
    return res

def write_xml(image_path,label_path,n,i,size,res):
    writer = PascalVocWriter(label_path,"../" + image_path + "/" + str(int(i)).zfill(5) + ".jpg",size,'Unknown',"../" + image_path + "/" + str(int(1)).zfill(5) + ".jpg")
    if(score):
        writer.addscore(score[0],score[1])
    for r in res:
        yolo_xcen = r[3][0]
        yolo_ycen = r[3][1]
        yolo_xh = r[3][2]
        yolo_yh = r[3][3]
        xml_xmin = int(yolo_xcen - yolo_xh/2.0)
        xml_ymin = int(yolo_ycen - yolo_yh/2.0)
        xml_xmax = int(yolo_xcen + yolo_xh/2.0)
        xml_ymax = int(yolo_ycen + yolo_yh/2.0)
        difficult = 0
        writer.addBndBox(xml_xmin,xml_ymin,xml_xmax,xml_ymax,r[1],r[2],difficult)
    writer.save(label_path + "/n" + str(n) + "_" + str(int(i)).zfill(5) + ".xml")

if __name__ == "__main__":
    net = load_net(cfg_path, weights_path, 0)
    meta = load_meta(meta_data_path)
    
    size = (640,480,3)
    lines = []
    score = []
    dobjs = []
    MOVIE_N_MAX = 1400
    NUM_IMAGES_MAX = 900

    for n in range(MOVIE_N_MAX):
        for i in range(NUM_IMAGES_MAX):
            scores = []
            dobjs = []
            image_file = xml_dir_path + str(n) + "_" + str(int(i)).zfill(5) + ".jpg"
            if(not(os.path.isfile(image_file))):
                 continue
            print(n,i)
            r = detect(net, meta, image_file)
            write_xml(image_file,label_path,n,i,size,r)
