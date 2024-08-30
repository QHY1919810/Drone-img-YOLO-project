from pyproj import Proj, transform
import numpy as np
import shapefile
import rasterio
import pandas as pd
import cv2
from pathlib import Path
from utils import *
import random
import os
csv_file = 'W:\\EDM\RSGIS\\gcastill\\DATA\\Cynthia\\20230816\\DJI_202308161037_all_CynthiaP1B50mm75magl85s85s-no_exif.csv'
dsm_path = 'D:\\hqiu_project\F8\\20230816_Cynthia_P1B50mm100magl85s85f-no_5cmDSM.tif'
camera_path = "D:\\hqiu_project\\F8\\20230816_Cynthia_P1B50mm100magl85s85f-no_cameras.txt"
cam_data = open(camera_path, "r")
bbox_project_path = 'W:\\EDM\\RSGIS\\hqiu\\droneimg_project_back_adjustedDataset\\'
#bbox_project_path = 'D:\\YOLOV8_orthomosaic_Aug2024\\drone_project_back\\'
#bboxfolder_path = 'D:\\hqiu_project\\train_label\\'
bboxfolder_path = 'W:\\EDM\\RSGIS\hqiu\\yolo_predict_droneadjusted\\'
cls_dict = {0:'poplar', 1:'spruce', 2:'pine', 3:'larch', 4:'snag'}
shp_file_name = 'predicted_project_back_AdjustedDataset'
img_data = pd.read_csv(csv_file)




cropped_width = 2048
cropped_height = 2048
image_width = 8192
image_height = 5460


P3857 = Proj(init='epsg:2955')
P4326 = Proj(init='epsg:4326')

pix_x_size = 4.39 * 10 **(-6)
pix_y_size = 4.39 * 10 **(-6)
f = 11018.8
cx = -10.8655
cy = 23.1932
b1 = 0
b2 = 0
k1 = -0.00584836
k2 = -0.274308
k3 = -0.417769
k4 = 0
p1 = -0.000460076
p2 = 0.000393682
p3 = 0
p4 = 0
bound = 10
# load dsm
dsm = rasterio.open(dsm_path)
dsm_arr = dsm.read(1)
try: dsm_epsg = int(str(dsm.crs)[5:])
except: exit("Error: DSM does not seem to have CRS information. Chcek and try again")

cfg = read_cfg("D:\\hqiu_project\\default.yml")

show_box = cfg['img2gis'].get('show_box',False)
coarse_del_Z = cfg['img2gis'].get('coarse_height_increment', 0.5) # starting increment for H in coarse to fine strategy of ray tracing/dsm interception
fine_del_H = cfg['img2gis'].get('fine_height_increment', 0.01)     # starting increment for H in coarse to fine strategy of ray tracing/dsm interception
coarse_tolerance = cfg['img2gis'].get('coarse_tolerance', 2)      # when we want to move to finer increments
fine_tolerance = cfg['img2gis'].get('fine_tolerance', 0.2)        # how close we want to approximate (±value is tolerable) 
bound = cfg['img2gis'].get('boundary', 5)                         # get dsm value as median in a small bxb region centered at px and py and curr_H:


count = 0
cameraData = []
camindexes = []
camcolumns = []
#create dataframe for the camera data with each photos
for line in cam_data:
    line = line.strip()
    
    if count == 0:
        splited = line.split(',')
        camcolumns = splited[1 : :]

    else:
        splited = line.split()
        camindexes.append(splited[0])
        cameraData.append(np.array(splited[1 : :]))
    count += 1 
cameraData = np.array(cameraData)
camDf = pd.DataFrame(cameraData, index=camindexes, columns=camcolumns)
output_file_projected = bbox_project_path + shp_file_name
w = shapefile.Writer(output_file_projected, shapefile.POLYGON)
w.field('source image', 'C', size=250)
w.field('confidence', 'C', size=250)
w.field('prediction', 'C', size=250)


for i in range(len(img_data)):
    
    img_path = img_data.loc[i, 'SourceFile']
    splitimgpath = img_path.split("/")
    curr_img_name = splitimgpath[-1]
    #remove .jpg
    curr_img_name_tmp = curr_img_name.split(".")
    curr_img_name = curr_img_name_tmp[0]
    curr_label_file_name = ''
    for fname in os.listdir(bboxfolder_path):
        if curr_img_name in fname:
            curr_label_file_name = fname

            print(fname)
            lat = img_data.loc[i, 'GpsLatitude']
            lon = img_data.loc[i, 'GpsLongitude']


            x,y = transform(P4326, P3857, lon, lat)
            exif_dict = extract_exif(img_path)
            altitude = float(exif_dict['Relative Altitude'])              # increase -> box bigger
            image_height = float(exif_dict['Exif Image Height'])
            image_width = float(exif_dict['Exif Image Width'])
            focal = float(exif_dict['Focal Length'].split(' ')[0]) / 1000 # increase -> box bigger
            sw = image_width * pix_x_size
            sh = image_height * pix_y_size



            omega = float(camDf.loc[curr_img_name]['Omega'])
            phi = float(camDf.loc[curr_img_name]['Phi'])
            kappa = float(camDf.loc[curr_img_name]['Kappa'])
            m = rot_matrix(kappa, omega, phi)
            Z0 = float(camDf.loc[curr_img_name]['Z'])
            X0 = float(camDf.loc[curr_img_name]['X'])   
            Y0 = float(camDf.loc[curr_img_name]['Y'])
            k = np.array([[f+b1, b2,  cx+image_width//2],
                        [   0,  f, cy+image_height//2],
                        [   0,  0,                 1]])
            k_inv = np.linalg.inv(k)

            
            

            
            bboxfile_path = bboxfolder_path + curr_label_file_name
            #print(curr_label_file_name)
            tmpoutpath = Path(bboxfile_path)
            if not tmpoutpath.is_file():
                continue
            anno_data = open(bboxfile_path, "r")
            bboxes = []
            classes = []
            confs = []


            for line in anno_data:
                line = line.strip()
                splited = line.split()
                bboxes.append(np.array(splited[1 : 5]))
                classes.append(np.array(splited[0]))
                confs.append(np.array(splited[5]))
                


            for j in range(len(bboxes)):
                bbox = np.float64(bboxes[j])
                conf = confs[j]
                cls = classes[j]
                
                bboxcx = (bbox[0]) * cropped_width
                bboxcy = (bbox[1]) * cropped_height
                boxw = bbox[2] * cropped_width
                boxh = bbox[3] * cropped_height
                minix = bboxcx - boxw/2
                maxix = bboxcx + boxw/2
                miniy = bboxcy - boxh/2
                maxiy = bboxcy + boxh/2

                pts = np.array([[bboxcx, bboxcy], [minix, miniy], [maxix, miniy], [maxix, maxiy],[minix, maxiy]])
                
                original_center = np.array([image_width//2+cx,image_height//2+cy])
                cropped_center = np.array([cropped_width//2+cx,cropped_height//2+cy])
                pts = pts - cropped_center + original_center

                full_extent = np.array([[image_width//2+cx,image_height//2+cy],[0,0],[image_width,0],[image_width,image_height],[0,image_height]])
                def iterate_undistort(points):
                    points = points.reshape(-1,2)
                    points = np.hstack([points, np.ones((len(points),1))])
                    xy_prime = k_inv.dot(points.T)[:2,:]

                    r = np.sqrt((xy_prime[0]**2 + xy_prime[1]**2))  # r_prime
                    # x_0 and y_0 init -> initial undistorted image coordinates
                    x = xy_prime[0] - xy_prime[0] * (k1 * r**2 + k2 * r**4 + k3 * r**6 + k4 * r**8) - p1 * (r**2 + 2 * xy_prime[0]**2) - 2 * p2 * xy_prime[0] * xy_prime[1] 
                    y = xy_prime[1] - xy_prime[1] * (k1 * r**2 + k2 * r**4 + k3 * r**6 + k4 * r**8) - p2 * (r**2 + 2 * xy_prime[1]**2) - 2 * p1 * xy_prime[0] * xy_prime[1] 
                    # iterate to get x,y -> final undistorted image coordinates
                    for _ in range(10):
                        r = np.sqrt((x**2 + y**2))
                        x = xy_prime[0] - x * (k1 * r**2 + k2 * r**4 + k3 * r**6 + k4 * r**8) - p1 * (r**2 + 2 * x**2) - 2 * p2 * x * y
                        y = xy_prime[1] - y * (k1 * r**2 + k2 * r**4 + k3 * r**6 + k4 * r**8) - p2 * (r**2 + 2 * y**2) - 2 * p1 * x * y

                    return x,y
            
                x,y = iterate_undistort(pts)
                xf,yf = iterate_undistort(full_extent)
                # get limits, scale xy --> sensor space location of centroid & principal point (m: meters)
                xmin, xmax = min(xf), max(xf)
                ymin, ymax = min(yf), max(yf)
                xf_m = (xf-xmin)/(xmax-xmin) * sw  # f: indicates full extent
                yf_m = (yf-ymin)/(ymax-ymin) * sh
                x_m = (x-xmin)/(xmax-xmin) * sw
                y_m = (y-ymin)/(ymax-ymin) * sh

                # ---------- Depth estimation through ray tracing ------------ #
                # Take centroid ray from its sensor location to lens, forward project it incremently until it 'intersects' DSM 
                # The following diagram and variable naming is used.
                # First we compute d' from known f and computed xy' distance (distance of centroid to principal point in meters in sensor space location)
                # Then, we increment curr_H, compute d using similarity property, and use this d to project the point and find its XYZ location (3D image coordinates in image space)
                # We convert this to UTM, and use the UTM coordinates to index the DSM and obtain the actual H (vertical distance to camera) at the computed (XY) point
                # If this distance is close enough to the current estimate of the distance, we stop as the current d is close enough to the actual distance from the camera to the centroid.
                # Otherwise, repeat with a larger curr_H.
                #   This can be thought of as tracing the ray in the direction of the centroid until it intercepts the DSM, with the goal of finding the length of the ray that just meets the DSM.

                # +-xy'-+                    [sensor plane]
                #  \    | 
                # d'\   | f
                #    \  |
                #     \ |
                #      \|
                # ------+------              [lens]
                #       |\
                #       | \
                #       |  \
                #       |   \
                #curr_Zc|    \  d (depth)     NOTE: curr_Zc (depth of principal point) roughly equal to curr_H (height above point in 3D world space)
                #       |     \                     for nadir images. For oblique, need to compute curr_H from curr d and 3d world distance b/n camera center and polygon centroid
                #       |      \
                #       |       \
                #       +---xy---+           [plane at Z=curr_H in image local coords]

                # compute xy' (undist_lengths_m) and d' (d_dash) in meters
                undist_lengths_m = np.sqrt((x_m-xf_m[0])**2 + (y_m-yf_m[0])**2) 
                d_dash_m = np.sqrt(undist_lengths_m**2 + focal**2)


                # start 
                del_Z = coarse_del_Z
                curr_Zc = 20 # starting estimate for Z (focal length in meters)
                while True:
                    # increment curr_H and compute curr_D
                    curr_Zc += del_Z
                    curr_d = curr_Zc * d_dash_m[0] / focal # [0] refers to first point (centroid)

                    # image local 3D coordinates for centroid (meters) : X = x*Z (Z:depth)
                    Xc = -x[0]*curr_d  # [0] refers to first point (centroid)
                    Yc =  y[0]*curr_d  

                    # convert local to global coordinate space (UTM) using rotation matrix m 
                    points = np.vstack([Xc,Yc,curr_Zc]).T
                    curr_poly = points.dot(m.T)
                    curr_poly = curr_poly[:,:2] / curr_poly[:,2:3]
                    curr_poly = -curr_d * curr_poly + np.array([X0, Y0])

                    # calculate elevation above point in 3D world space:
                    dist_m_sq = (X0-curr_poly[0][0])**2 + (Y0-curr_poly[0][1])**2 # UTM distance between centroid and camera center
                    curr_H = np.sqrt(curr_d**2 - dist_m_sq)

                    # calculate pixel coordinate in dsm raster (of center)
                    py, px = dsm.index(curr_poly[0][0], curr_poly[0][1])

                    # get dsm value as median in a small bxb region centered at px and py and curr_H:
                    slice_ = dsm_arr[py-bound:py+bound,px-bound:px+bound]
                    dsm_value = np.median(slice_[slice_ > 0])
                    if np.isnan(dsm_value):
                        print(slice_, np.unique(slice_))
                        print("Error! No valid values in projected location. Please make sure DSM is for the correct location")
                        break
                    
                    # get distance to camera according to DSM at projected ray's location (curr_H') 
                    # and compute difference with curr_H. If diff < tolerance, the current ray has succesfully intercepted the DSM 
                    computed_del_H = Z0 - dsm_value # Z0: absolute altitude of drone over sea level. DSM also has absolute altitudes. Both in m
                    diff = computed_del_H - curr_H
                    
                    if abs(diff) < fine_tolerance:
                        break
                    
                    elif diff < 0:
                        # error if we overshot the DSM (and landed up inside) - need to retry 
                        print(f"Computed H => {computed_del_H}, Actual => {curr_H}")
                        print("Error! Too coarse. Reduce height increment del_H or increase tolerance, and try again")     
                        break

                    elif diff < coarse_tolerance: 
                        # reduce the height incrmeent (start finer approximation)
                        del_Z = fine_del_H 
                # -------------- Actual Projection -------------- #
                # Now that we have the point depth of the centroid, we need to compute the depths of all actual points
                # using the estimated value of curr_H. xy*Z = XY (local coordinate space)
                # NOTE: alternatively, we can reuse the projection idea from above, but it may not be as robust and will be slower (by a bit). 

                corrected_point_depths = curr_Zc * d_dash_m / focal
                X = -x*corrected_point_depths
                Y =  y*corrected_point_depths # NOTE: may have to switch the - around

                # append focal length
                f_col = np.array([curr_Zc]*len(X))    # f should be curr_H here (in 3D image coordinate system, all are assumed to be on the same plane (but natrually at different depths from the camera))
                points = np.vstack([X,Y, f_col]).T

                # calculate new polygon points (in global coord system)
                curr_poly = points.dot(m.T)
                curr_poly = curr_poly[:,:2] / curr_poly[:,2:3]
                curr_poly = -corrected_point_depths.reshape(-1,1) * curr_poly + np.array([X0, Y0])

                # compute distance to image center (only used for batch shapefile creation)
                dist = np.sqrt((X0-curr_poly[0][0])**2 + (Y0-curr_poly[0][1])**2)

                #print(curr_poly)
                w.poly([curr_poly.tolist()[1:]])
                w.record(img_path, str(conf), cls_dict[int(cls)]) 
                print(cls_dict[int(cls)], conf,img_path)


                

w.close()
wkt = getWKT_PRJ(dsm_epsg)
prj = open(f"{output_file_projected}.prj", "w")
prj.write(wkt) # TODO: config file for EPSG (only do this if None)
prj.close()
                    

