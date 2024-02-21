import ee
import geemap
from osgeo import gdal
import numpy as np
import rasterio 
import json
import streamlit as st

json_data = st.secrets["json_data"]

# Preparing values
json_object = json.loads(json_data, strict=False)
service_account = json_object['client_email']
json_object = json.dumps(json_object)

# Authorising the app
credentials = ee.ServiceAccountCredentials(service_account, key_data=json_object)
ee.Initialize(credentials)

# ee.Initialize()

platform = 'projects/planet-nicfi/assets/basemaps/africa'

Pl = (ee.ImageCollection(platform)
      .select(['B','G','R','N']))

ee.Initialize()

def get_image(coordinates, radius, what_img, model, year = '2016'):
    
    platform = 'projects/planet-nicfi/assets/basemaps/africa'

    poi = ee.Geometry.Point(coordinates)
    roi = poi.buffer(radius).bounds()
    
    Pl = (ee.ImageCollection(platform)
          .select(['B','G','R','N']))
    
    if what_img == 'latest':
        size = Pl.size()
        Pl_list = Pl.toList(size)
        img = ee.Image(Pl_list.get(size.getInfo()-1))
        clipped = img.clip(roi)
    elif what_img == 'median':
        # ee.Image(Pl_gdf).setDefaultProjection(epsg)
        if (model == 'Target-only') | (model == 'DANN'):
            clipped = ee.Image(Pl.filterDate(year+'-01-01', year+'-12-31').median().clip(roi)).setDefaultProjection('EPSG:4236')
        else:
            clipped = ee.Image(Pl.filterDate(year+'-01-01', year+'-12-31').median().clip(roi)).setDefaultProjection('EPSG:4236')
    
    geemap.download_ee_image(clipped, "test/clipped.tif", scale=4.77)
    
    return clipped

def crop_image(filename = 'test/clipped.tif', out_filenames = 'test/crop_', patch_size = 256, overlap = 0):

    im = gdal.Open(filename)
    k = 0
    s = '{:0'+str(3)+'d}'
    for i in np.arange(0, im.RasterXSize, int(patch_size*(1-overlap))):
        for j in np.arange(0, im.RasterYSize, int(patch_size*(1-overlap))):
            gdal.Translate(out_filenames + s.format(k)+'.tif', im, srcWin = [i, j, patch_size, patch_size])

            with rasterio.open(out_filenames + s.format(k)+'.tif') as src:
                img = src.read()
                perc = (np.sum(sum(img != 0) == 4) > patch_size*patch_size*0.95) & (np.sum(np.isnan(img)[0]) < 0.1*256*256)
                src.close()

            if perc:
                k+=1

    im = None