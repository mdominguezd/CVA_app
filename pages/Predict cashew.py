import streamlit as st
import folium
from folium.plugins import Draw
import contextlib
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
import sys
import pathlib

import os

# This adds the path of the …/src folder
# to the PYTHONPATH variable
sys.path.append("/mount/src/cva_app/")

from get_image import get_image, crop_image
from DL_backend import Img_Dataset, predict_cashew

st.set_page_config(page_title="CVA", page_icon=":deciduous_tree:")

st.title(':deciduous_tree: CashewVisionAdapt (CVA) :satellite:')

st.header('Select Parameters:')

model = st.radio('Model:', ['Source-only', 'Target-only', 'DANN'])

planet_img = st.radio('Select a planet image:',['median', 'latest'])

if planet_img == 'median':
    year = st.slider('Year:',2015, 2022, 2018, step = 1)
    year = str(year)
else:
    year = '2024'

st.markdown('#### Draw a marker in the area of the map where you want to predict cashew crops and click the RUN button:')

run = st.button('RUN')

m = folium.Map(location = [7,10], zoom_start = 3)
tile = folium.TileLayer(
        tiles = 'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = True,
        control = True
       ).add_to(m)

tile = folium.TileLayer(
        tiles = 'https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = True,
        control = True
       ).add_to(m)

# folium.GeoJson('../Data/Vector/Cashew_Polygons_TNZ_splitted_KM.geojson',
#                tooltip=folium.GeoJsonTooltip(fields=['split'])
#               ).add_to(m)

# folium.GeoJson('../Data/Vector/Cashew_Polygons_CIV_splitted_KM.geojson',
#                tooltip=folium.GeoJsonTooltip(fields=['split'])
#               ).add_to(m)
    
Draw(export = False, draw_options = {'polyline' : False, 'polygon': False, 'rectangle' : False, 'circle' : False, 'circlemarker' : False}).add_to(m)

map = st_folium(m, width = 700, height = 500, returned_objects = ['last_active_drawing'])

if map['last_active_drawing'] != None:

    if run:

        with st.spinner('Gathering planet images from Google Earth Engine and predicting Cashew crops with '+model+' model...'):
        
            coordinates = list(map['last_active_drawing']['geometry']['coordinates'])
            
            with contextlib.suppress(PermissionError):
                im = get_image(coordinates, radius = 2000, what_img = planet_img, model = model, year = year)
                
                crop_image()
        
            DS = Img_Dataset('test', norm = 'Linear_1_99', VI = True, domain = 'target')        
        
            domain = predict_cashew(DS, model)

            st.write('Domain predicted:' + domain)