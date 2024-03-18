import folium
import streamlit as st
from streamlit_folium import st_folium
from folium.plugins import Draw

st.set_page_config(page_title="CVA", page_icon=":deciduous_tree:", initial_sidebar_state="collapsed")

st.markdown("""
# :deciduous_tree: CashewVisionAdapt (CVA) :satellite:

Thanks for helping improve the baseline information of cashew crop locations in Africa!

Draw some polygons, export them and send them to us so we can improve the prediction models.

E-mail: <martin.dominguezduran@wur.nl>

""")

domain = st.radio('Select domain', ['Ivory Coast', 'Tanzania'])

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
    
Draw(export = True, 
     filename = domain + '_labels.geojson',
     draw_options = {'polyline' : False, 'polygon': True, 'rectangle' : False, 'circle' : False, 'circlemarker' : False, 'marker' : False}).add_to(m)

map = st_folium(m, width = 700, height = 500, returned_objects = ['last_active_drawing'])