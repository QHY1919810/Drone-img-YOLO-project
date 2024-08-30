To generate the dataset, there are several steps:

Step1:
In mappingbox.py 
moddify the following:
shpfilepath = 'the shape file that containing the bounding boxes that you want to project to image'
csv_file = 'the csv file that containing the records of informations like latitude, longitude, etc. when the drone photo is taken'
Then, type python mappingbox.py, it will find the four closest drone images of the box, and mapShape2img.txt will record these information.

Step2:
In mapping_coords.py
moddify the following:
shpfilepath = 'the shape file that containing the bounding boxes that you want to project to image'
csv_file = 'the csv file that containing the records of informations like image path, latitude, longitude, etc. when the drone photo is taken'
dsm_path = 'the dsm file of the forest'
camera_path = 'the txt file of the camera data of the drone photos like roll, pitch, yaw, etc. at the time the photo is taken'
name_dict = 'The number represent each type of tree in yolo model, for example {'poplar':0, 'spruce':1, 'pine':2, 'larch':3, 'snag':4}'
outputImgpath = 'The folder that contain the output image'
outputLabelpath = 'The folder that contain the output yolov8 txt file'
cropped_width = 'The width of the cropped image'
cropped_height = 'The height of the cropped image'

Then, type python mapping_coords.py, it will project the bounding boxes to those image, and crop the image to 2048x2048, then generate yolov8
txt annotation file with these cropped images, then we upload those images and annotations to Roboflow so that we can adjust the bounding boxes to
be more precisely. 

Step3:
In project_img_footprint.py
csv_file = 'the csv file that containing the records of informations like image path, latitude, longitude, etc. when the drone photo is taken'
dsm_path = 'the dsm file of the forest'
camera_path = 'the txt file of the camera data of the drone photos like roll, pitch, yaw, etc. at the time the photo is taken'
cropped_width = 'The width of the cropped image'
cropped_height = 'The height of the cropped image'

Then, type python project_img_footprint.py, this code will project the four corners and the centroids(which is the camera principle point) of the 
2048x2048 imgs to the GIS shape file, so that other people can label those shape files which seperate them into train, trainVal and testing

Step4:
In map_drone_img.py
After we manually adjusted the boxes and then download the adjusted dataset from roboflow as well as finish seperating the train, trainVal and testing part 
of dataset in the shape file, we can use the part of code.
Note: the shpfilepath in this file means the footprint that is already labelled with 
shpfilepath = 'W:\\EDM\\RSGIS\\hqiu\\miscs\\YOLOV8_orthomosaic_Aug2024\\Cynthia20240816_2048centralFootprints_EPSG2955_SR_edited.shp'

Images source and labels source path
imagesrcpath = 'W:\\EDM\\RSGIS\\hqiu\\miscs\\YOLOV8_orthomosaic_Aug2024\\Drone Image Tree Detection.v5i.yolov8\\train\\images'
labelsrcpath = 'W:\\EDM\\RSGIS\\hqiu\\miscs\\YOLOV8_orthomosaic_Aug2024\\Drone Image Tree Detection.v5i.yolov8\\train\\labels'

The path to store images and labels for the train validation set
trainval_imgs = 'W:\\EDM\\RSGIS\\hqiu\\miscs\\YOLOV8_orthomosaic_Aug2024\\final_drone_dataset\\trainVal\\images'
trainval_labels = 'W:\\EDM\\RSGIS\\hqiu\\miscs\\YOLOV8_orthomosaic_Aug2024\\final_drone_dataset\\trainVal\\labels'

The path to store images and labels for the train set
train_imgs = 'W:\\EDM\\RSGIS\\hqiu\\miscs\\YOLOV8_orthomosaic_Aug2024\\final_drone_dataset\\train\\images'
train_labels = 'W:\\EDM\\RSGIS\\hqiu\\miscs\\YOLOV8_orthomosaic_Aug2024\\final_drone_dataset\\train\\labels'

The path to store images and labels for the testing set
testing_imgs = 'W:\\EDM\\RSGIS\\hqiu\\miscs\\YOLOV8_orthomosaic_Aug2024\\final_drone_dataset\\testing\\images'
testing_labels = 'W:\\EDM\\RSGIS\\hqiu\\miscs\\YOLOV8_orthomosaic_Aug2024\\final_drone_dataset\\testing\\labels'
