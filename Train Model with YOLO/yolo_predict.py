from ultralytics import YOLO
import cv2
import os
import numpy as np

#The file paths
model_path = 'W:\\EDM\\RSGIS\\hqiu\\yolo_model_drone_final\\weights\\best.pt'
testingimg_path = 'D:\\YOLOV8_orthomosaic_Aug2024\\final_drone_dataset\\testing\\images'
testingpredict_path = 'W:\\EDM\\RSGIS\\hqiu\\yolo_predict_droneadjusted'

model = YOLO(model_path)
for f in os.listdir(testingimg_path):
    img_name = f.split(".")
    img_name = img_name[0]
    curr_img_path = testingimg_path + '\\' + f
    curr_img = cv2.imread(curr_img_path)
    curr_txt_path = testingpredict_path + '\\' + img_name + '.txt'
    results = model.predict(source = curr_img, save=True)
    print(curr_img_path)
    r = results[0]
    confs = r.boxes.conf
    classes = r.boxes.cls
    boxes_annotation = r.boxes.xywhn
    curr_output_label = open(curr_txt_path, "w")
    for i in range(len(classes)):
        curr_conf = confs[i]
        curr_cls = int(classes[i])
        curr_box = boxes_annotation[i]
        curr_output_label.write(str(curr_cls)+ ' ' + str(float(curr_box[0]))+ ' ' + str(float(curr_box[1]))+ ' ' + str(float(curr_box[2]))+ ' ' + str(float(curr_box[3]))+ ' ' + str(float(curr_conf))+ '\n')
    curr_output_label.close()

        