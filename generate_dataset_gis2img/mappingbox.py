from pyproj import Proj, transform
from dbfread import DBF
import numpy as np
import pandas as pd
import shapefile
#The file paths
shpfilepath = 'W:\\EDM\RSGIS\\gcastill\\DATA\\Cynthia\\20230816\\YOLOforestsat\\cynthia20230816_ReferenceBoxes_v20240801.shp'
csv_file = 'W:\\EDM\RSGIS\\gcastill\\DATA\\Cynthia\\20230816\\DJI_202308161037_all_CynthiaP1B50mm75magl85s85s-no_exif.csv'


f = open("mapShape2img"+".txt", "w")
sf = shapefile.Reader(shpfilepath)
sp = sf.shapes()
img_path_dict = {}
addrs = []
img_data = pd.read_csv(csv_file)

P3857 = Proj(init='epsg:2955')
P4326 = Proj(init='epsg:4326')
for i in range(len(img_data)):
    name = img_data.loc[i, 'SourceFile']
    lat = img_data.loc[i, 'GpsLatitude']
    lon = img_data.loc[i, 'GpsLongitude']
    x,y = transform(P4326, P3857, lon, lat)
    addrs.append([i , x, y ])



for i in range(len(sp)):
    tmpbbox = sp[i].bbox
    tmpcentre = [(tmpbbox[0] + tmpbbox[2]) / 2 , (tmpbbox[1] + tmpbbox[3]) / 2 ]
    dists = []
    for addr in addrs:
        tmpcoord = [addr[1],addr[2] ]
        tmpdist = ((tmpcoord[0] - tmpcentre[0])**2 + (tmpcoord[1] - tmpcentre[1])**2)
        dists.append(tmpdist)
    sortedargs = np.argsort(np.array(dists))
    print(i)
    f.write(str(i)+' '+str(addrs[sortedargs[0]][0])+' '+str(addrs[sortedargs[1]][0])+' '+str(addrs[sortedargs[2]][0])+' '+str(addrs[sortedargs[3]][0])+' '+'\n')
f.close()

