#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np  
import sys,os  
import cv2
caffe_root = '/home/bu5/bu5project/caffe_yolov2/'
sys.path.insert(0, caffe_root + 'python')  
import caffe

import detect_tool as tool

CLASSES = ["hold", "stop", "shutter"]

def draw_box(pic_name, boxes):
    image = cv2.imread(pic_name)
    img_h = image.shape[0]
    img_w = image.shape[1]
    print("image:", image.shape)
    for box in boxes:
        x = box.rect.corner.x * img_w
        y = box.rect.corner.y * img_h
        w = box.rect.width    * img_w
        h = box.rect.height   * img_h
        confidence = box.prob
        classes    = box.category
        # print(box.rect.corner.x, box.rect.corner.y, box.rect.width, box.rect.height)
        # print(classes, confidence, x, y, w, h)
        
        left   = max(0, min(int(x - w/2), img_w - 1));
        top    = max(0, min(int(y - h/2), img_h - 1));
        right  = max(0, min(int(x + w/2), img_w - 1));
        bottom = max(0, min(int(y + h/2), img_h - 1));
        print(classes, confidence, left, top, right, bottom)
        
        text = CLASSES[classes]
        cv2.rectangle(image, (left, top), (right, bottom), (255,0,255), 3)
        cv2.putText(image, text, (left+5, top), cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,255), 5);
    
    name = "result-" + pic_name.split("/")[-1]
    cv2.imwrite(name, image)
    print(pic_name.split("/")[-1])
    print("Draw box completed!")
    print("**********************")
    
    return 0

def detect_yolov2(pic_name):
    # data
    image = caffe.io.load_image(pic_name) # 使用caffe接口caffe.io.load_image()读图片，是RGB格式，scale在0～1之间的float。
    # image = cv2.imread(pic_name) # 使用opencv读进来的图片，是BGR格式，0～255，通道格式为（h,w,c），即（row,col,channel）
   
    transformer = caffe.io.Transformer({'data': (1, 3, 416, 416)})
    transformer.set_transpose('data', (2, 0, 1)) 
    transformed_image = transformer.preprocess('data', image)
    print(transformed_image.shape)  
        
    net.blobs['data'].data[...] = transformed_image
    output = net.forward()
    
    feat = net.blobs['region1'].data[0]
    print(feat.shape)

    # 配置region层参数
    boxes_of_each_grid = 5
    classes = 3
    thread  = 0.4
    # biases  = np.array([0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828])
    biases = np.array([1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]) # voc

    # get boxes
    boxes = tool.get_region_boxes(feat, boxes_of_each_grid, classes, thread, biases)
    print("Detection completed!")
    
    # 画box
    draw_box(pic_name, boxes)
    
    
if __name__ == '__main__':
    # 配置路径
    model_def     = 'yolov2_tiny_3.prototxt'
    model_weights = 'yolov2_tiny_3.caffemodel' 
    test_dir      = "gesture_recognition_test.txt"

    # 加载模型
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    
    ft = open(test_dir,'r')
    ft_line = ft.readline()

    i = 1
    while ft_line:  

        if i > 10:
            exit()
        ft_line = ft_line.strip('\n')
        imgfile = "%s"%(ft_line)
        # print(imgfile)
        
        # 检测目标
        detect_yolov2(imgfile)
        
        ft_line = ft.readline()
        
        i += 1
    ft.close()
    
    
