import shapefile
import numpy as np
import os
import shutil
#The file paths 
shpfilepath = 'W:\\EDM\\RSGIS\\hqiu\\miscs\\YOLOV8_orthomosaic_Aug2024\\Cynthia20240816_2048centralFootprints_EPSG2955_SR_edited.shp'
imagesrcpath = 'W:\\EDM\\RSGIS\\hqiu\\miscs\\YOLOV8_orthomosaic_Aug2024\\Drone Image Tree Detection.v5i.yolov8\\train\\images'
labelsrcpath = 'W:\\EDM\\RSGIS\\hqiu\\miscs\\YOLOV8_orthomosaic_Aug2024\\Drone Image Tree Detection.v5i.yolov8\\train\\labels'
trainval_imgs = 'W:\\EDM\\RSGIS\\hqiu\\miscs\\YOLOV8_orthomosaic_Aug2024\\final_drone_dataset\\trainVal\\images'
trainval_labels = 'W:\\EDM\\RSGIS\\hqiu\\miscs\\YOLOV8_orthomosaic_Aug2024\\final_drone_dataset\\trainVal\\labels'
train_imgs = 'W:\\EDM\\RSGIS\\hqiu\\miscs\\YOLOV8_orthomosaic_Aug2024\\final_drone_dataset\\train\\images'
train_labels = 'W:\\EDM\\RSGIS\\hqiu\\miscs\\YOLOV8_orthomosaic_Aug2024\\final_drone_dataset\\train\\labels'
testing_imgs = 'W:\\EDM\\RSGIS\\hqiu\\miscs\\YOLOV8_orthomosaic_Aug2024\\final_drone_dataset\\testing\\images'
testing_labels = 'W:\\EDM\\RSGIS\\hqiu\\miscs\\YOLOV8_orthomosaic_Aug2024\\final_drone_dataset\\testing\\labels'

sf = shapefile.Reader(shpfilepath)
sp = sf.shapes()
sr = sf.records()
imgfiles = os.listdir(imagesrcpath)
#print(imgfiles)
testing = []
validation = []
training = []
for i in range(len(sr)):
    img_path = sr[i][0]
    type = sr[i][1]
    splitimgpath = img_path.split("/")
    curr_img_name = splitimgpath[-1]
    curr_img_name_tmp = curr_img_name.split(".")
    curr_img_name = curr_img_name_tmp[0]
    #print(curr_img_name)
    for n in imgfiles:
        if curr_img_name in n :
            curr_n_tmp = n.split(".")
            curr_n = ''
            for tmp in range(len(curr_n_tmp) - 1):
                curr_n += str(curr_n_tmp[tmp])
                curr_n += '.'
            curr_txt = curr_n + 'txt'
            curr_jpg = curr_n + 'jpg'
            print(curr_img_name)
            txt_src = labelsrcpath + '\\' + curr_txt
            jpg_src = imagesrcpath + '\\' + curr_jpg
            if type == 'training':
                txt_dst = train_labels + '\\' + curr_txt
                jpg_dst = train_imgs + '\\' + curr_jpg
            if type == 'testing':
                txt_dst = testing_labels + '\\' + curr_txt
                jpg_dst = testing_imgs + '\\' + curr_jpg
            if type == 'validation':
                txt_dst = trainval_labels + '\\' + curr_txt
                jpg_dst = trainval_imgs + '\\' + curr_jpg
            #print(txt_src, txt_dst)
            shutil.copyfile(txt_src, txt_dst)
            shutil.copyfile(jpg_src, jpg_dst)
            #print(jpg_src)