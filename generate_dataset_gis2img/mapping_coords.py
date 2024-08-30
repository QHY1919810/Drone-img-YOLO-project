from pyproj import Proj, transform
from dbfread import DBF
import numpy as np
import shapefile
import rasterio
import pandas as pd
import cv2
from pathlib import Path
from utils import *
import random
#The file paths
shpfilepath = 'W:\\EDM\RSGIS\\gcastill\\DATA\\Cynthia\\20230816\\YOLOforestsat\\cynthia20230816_ReferenceBoxes_v20240801.shp'
csv_file = 'W:\\EDM\RSGIS\\gcastill\\DATA\\Cynthia\\20230816\\DJI_202308161037_all_CynthiaP1B50mm75magl85s85s-no_exif.csv'
dsm_path = 'W:\\EDM\RSGIS\\gcastill\\DATA\\Cynthia\\20230816\\20230816_Cynthia_P1B50mm100magl85s85f-no_5cmDSM.tif'
camera_path = "W:\\EDM\\RSGIS\gcastill\\DATA\Cynthia\\20230816\\buffered_tiles\F8\\20230816_Cynthia_P1B50mm100magl85s85f-no_cameras.txt"
outputImgpath = 'W:\\EDM\RSGIS\\hqiu\\miscs\\ImgLabelCropped\\images\\'
outputLabelpath = 'W:\\EDM\RSGIS\\hqiu\\miscs\\ImgLabelCropped\\labels\\'
#The size of the cropped image
cropped_width = 2048
cropped_height = 2048


name_dict = {'poplar':0, 'spruce':1, 'pine':2, 'larch':3, 'snag':4}
cam_data = open(camera_path, "r")
shape2img = open("mapShape2img"+".txt", "r")

img_data = pd.read_csv(csv_file)

gen_label = 1
sf = shapefile.Reader(shpfilepath)
sp = sf.shapes()
sr = sf.records()

P3857 = Proj(init='epsg:2955')
P4326 = Proj(init='epsg:4326')

focal = 50 / 1000
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




for line in shape2img:
    splited = line.split(" ")
    curr_n = int(splited[0])
    tmpbbox = sp[curr_n].bbox
    tmppoints = sp[curr_n].points
    tree_type = name_dict[sr[curr_n][2]]
    print("current tree is: ",sr[curr_n][2], ', Label no. is ', tree_type)

    _tmppoints = np.array(tmppoints)
    tmpcentre = [(tmpbbox[0] + tmpbbox[2]) / 2 , (tmpbbox[1] + tmpbbox[3]) / 2 ]  #x,y
    poly0 = np.vstack([tmpcentre, np.flipud(_tmppoints)])
    color1 = (50,50,255)
    #lon,lat = transform(P3857, P4326, tmpcentre[0], tmpcentre[1])

    #modified from gis2img function of Image-GIS-Projection, Author: Rudraksh Kapil, July 2023, https://github.com/rudrakshkapil/Image-GIS-Projection/tree/main
    for img_idx in range(1,5):
        poly = poly0
        img_path = img_data.loc[int(splited[img_idx]), 'SourceFile']
        exif_dict = extract_exif(img_path)
        altitude = float(exif_dict['Relative Altitude'])              # increase -> box bigger
        image_height = float(exif_dict['Exif Image Height'])
        image_width = float(exif_dict['Exif Image Width'])
        focal = float(exif_dict['Focal Length'].split(' ')[0]) / 1000 # increase -> box bigger
        #print(image_height, image_width)
        sw = image_width * pix_x_size
        sh = image_height * pix_y_size
        # sw= 0.0360448        # ---> sensor_width       (in meters), can be calculated as sensor_width (in m/pixel) * img_width (in pixels)
        # sh= 0.024024         # ---> sensor_height      (in meters)
        splitimgpath = img_path.split("/")
        curr_img_name = splitimgpath[-1]
        #remove .jpg
        curr_img_name_tmp = curr_img_name.split(".")
        curr_img_name = curr_img_name_tmp[0]
        #print(curr_img_name)
        #print(camDf.loc[curr_img_name]['X'])
        omega = float(camDf.loc[curr_img_name]['Omega'])
        phi = float(camDf.loc[curr_img_name]['Phi'])
        kappa = float(camDf.loc[curr_img_name]['Kappa'])
        
        m = rot_matrix(kappa, omega, phi)
        Z0 = float(camDf.loc[curr_img_name]['Z'])
        X0 = float(camDf.loc[curr_img_name]['X'])   
        Y0 = float(camDf.loc[curr_img_name]['Y'])
        gps_lat = convert_gps(exif_dict["GPS Latitude"])
        gps_lon = convert_gps(exif_dict["GPS Longitude"])

        py, px = dsm.index(tmpcentre[0], tmpcentre[1])
        slice_ = dsm_arr[py-bound:py+bound,px-bound:px+bound]
        centroid_dsm_value = np.median(slice_[slice_ > 0])
        if np.isnan(centroid_dsm_value):
            print(slice_, np.unique(slice_))
            exit("Error! No valid values in projected location. Please make sure DSM is for the correct location")
        curr_H = Z0 - centroid_dsm_value
        # print(curr_H)
        



        # get distance to center of camera 
        XY_dist_m_sq = (poly[:,0] -X0)**2 + (poly[:,1]-Y0)**2

        # get depth 
        point_depths_m = np.sqrt(curr_H**2 + XY_dist_m_sq)

        poly = (poly - np.array([X0,Y0])) / (-point_depths_m).reshape(-1,1)
        poly = np.hstack([poly, np.ones((len(poly), 1))])
        pts = poly.dot(m)      # note: for m, inv == transpose NOTE: maybe remove .T

        pts = pts*-focal   # backproject 
        #print(pts)
        pts = pts[:,:2]    # remove last col (-focal)

        pts = pts / np.array([sw, sh])

        pts *= np.array([-(image_width+cx), -(image_height+cy)])
        pts += np.array([image_width/2, image_height/2]) 

        tmp0 = np.array(pts[:,0])
        tmpy = np.array([pts[:, 1]]).T
        tmp1 = np.ones(len(tmp0)) * (image_width/2)
        tmp2 = np.array([tmp1 + tmp1 - tmp0]).T
        new0 = np.hstack([tmp2, tmpy])

        pts = new0






        if (pts[0][0] >= 0 and pts[0][0] < image_width and pts[0][1] >= 0 and pts[0][1] < image_height):
            tmp_center_1 = pts[0].astype(np.uint64)
            print(tmp_center_1)
            
            outputname = outputImgpath + curr_img_name  + '_2048cropped' + '.png'
            outputlabelpath = outputLabelpath + curr_img_name  + '_2048cropped' + '.txt'
            
            outpath = Path(outputname)
            if outpath.is_file():
                t = 0
                #imgtmp = cv2.imread(outputname)

            else:
                imgtmp = cv2.imread(img_path)
                print("read new img \n")
                curr_img = np.array(imgtmp)
                curr_img = cv2.rotate(curr_img, cv2.ROTATE_90_CLOCKWISE)
                cropped = curr_img[int(image_width//2+cx - cropped_width//2):int(image_width//2+cx + cropped_width//2),int(image_height//2+cy - cropped_height//2):int(image_height//2+cy + cropped_height//2)]  
                imgtmp = cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)
                cv2.imwrite(outputname, imgtmp)

            original_center = np.array([image_width//2+cx,image_height//2+cy])
            cropped_center = np.array([cropped_width//2+cx,cropped_height//2+cy])
            pts = pts - original_center + cropped_center

            minix = np.min(pts[:, 0])
            miniy = np.min(pts[:, 1])
            maxix = np.max(pts[:, 0])
            maxiy = np.max(pts[:, 1])

            if (minix >= 0 and maxix < cropped_width and miniy >= 0 and maxiy < cropped_height):
                if gen_label == 1:
                    boxcx = ((minix + maxix) / 2 ) / cropped_width
                    boxcy = ((miniy + maxiy) / 2 ) / cropped_height
                    class_name = str(tree_type)
                    boxw = (maxix - minix) / cropped_width
                    boxh = (maxiy - miniy) / cropped_height
                    labelfile = open(outputlabelpath, "a")
                    labelfile.write(class_name+' '+str(boxcx)+' '+str(boxcy)+' '+str(boxw)+' '+str(boxh)+' '+'\n')
                    labelfile.close() 


                #cv2.polylines(imgtmp, [np.int32(pts[ 1 : :])], True, color=color1, thickness=2)
                label = "Shape No." + str(curr_n)
                #cv2.putText(imgtmp, label, (int(minix), int(miniy) - 15), cv2.FONT_HERSHEY_SIMPLEX, 2, color1, 5)
            
                    
            #cv2.imwrite(outputname, imgtmp)
            
    
    

    



