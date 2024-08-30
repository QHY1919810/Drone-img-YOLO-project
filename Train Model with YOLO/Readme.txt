###Where to find the dataset:
The dataset of orthomosaic images:
W:\EDM\RSGIS\hqiu\miscs\YOLOV8_orthomosaic_Aug2024\tiles_small_test
The dataset of drone images:
W:\EDM\RSGIS\hqiu\miscs\YOLOV8_orthomosaic_Aug2024\final_drone_dataset


1. To train the YOLO model, modify the parameters in config.yaml and data.yaml
and then type:
python -m torch.distributed.run --nproc_per_node=2 main.py --mode Train --config config.yaml

2.To use the Model to make prediction
In yolo_predict.py
model_path = 'the YOLO model trained before'
testingimg_path = 'The testing images to make prediction'
testingpredict_path = 'The folder that output the predictions'

Then type
python yolo_predict.py

The output will be in the format as
class cx cy width height confidence
The xcenter, ycenter ,width, height of the bounding box, and they are normalized with image size



3. Convert YOLO prediction to GIS shape file
In convertbbox2gis.py,
csv_file = 'the csv file that containing the records of informations like image path, latitude, longitude, etc. when the drone photo is taken'
dsm_path = 'the dsm file of the forest'
camera_path = 'the txt file of the camera data of the drone photos like roll, pitch, yaw, etc. at the time the photo is taken'
bbox_project_path = 'The path to the folder that contains the shape file'
bboxfolder_path = 'The folder that contains the yolo predictions to be convert'
shp_file_name = 'The name of the output shape file'

Then type 
python convertbbox2gis.py
The output will be a shape file that contain all the projected boxes from the folder that contains the yolo predictions


####Environment Path:
C:\anaconda3_public\anaconda3\envs\yolov8_haoyu

