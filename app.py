import streamlit as st
import folium
from folium.plugins import Draw
import contextlib
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
import sys
import pathlib

import U_Net, U_Net_DANN
from get_image import get_image, crop_image
from DL_backend import Img_Dataset, predict_cashew

st.set_page_config(page_title="CVA", page_icon=":deciduous_tree:")

st.title(':deciduous_tree: CashewVisionAdapt (CVA) :satellite:')

st.sidebar.header('Select Parameters:')

model = st.sidebar.radio('Model:', ['Source-only', 'Target-only', 'DANN'])

planet_img = st.sidebar.radio('Select a planet image:',['median', 'latest'])

if planet_img == 'median':
    year = st.sidebar.slider('Year:',2015, 2022, 2018, step = 1)
    year = str(year)

st.write('Predicting with model: ' + str(model))

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
    
Draw(draw_options = {'polyline' : False, 'polygon': False, 'rectangle' : False, 'circle' : False, 'circlemarker' : False}).add_to(m)

map = st_folium(m, width = 700, height = 500, returned_objects = ['last_active_drawing'])

if map['last_active_drawing'] != None:

    with st.spinner('Wait for it...'):
    
        coordinates = list(map['last_active_drawing']['geometry']['coordinates'])
        
        with contextlib.suppress(PermissionError):
            im = get_image(coordinates, radius = 2000, what_img = planet_img, model = model, year = year)
            
            crop_image()
    
        DS = Img_Dataset('test', norm = 'Linear_1_99', VI = True, domain = 'target')        
    
        predict_cashew(DS, model)
