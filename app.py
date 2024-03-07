import streamlit as st
from streamlit_extras.switch_page_button import switch_page 

st.set_page_config(page_title="CVA", page_icon=":deciduous_tree:", initial_sidebar_state="collapsed")

st.markdown("""
# Welcome to Cashew Vision Adapt (CVA) :deciduous_tree: :satellite:
""")

# Center image
left_co, cent_co,last_co = st.columns(3)

with cent_co:
    st.image('img/picture_.png')

st.markdown("""
In this dashboard you will be able to evaluate the performance of the models trained on any chosen year with planet imagery and at any area of interest in Africa using the `Predict cashew` page.

Additionally, we encourage you to label some cashew polygons on the `Label data` page, export them and send them to us so the predictions of the models can improve.
""")

col1, col2 = st.columns(2)

with col1:
    predict = st.button("Predict cashew :deciduous_tree:")
    
    if predict:
        switch_page("Predict cashew")

with col2:
    label = st.button("Label data :black_square_button: :pencil2:")
    
    if label:
        switch_page("Label data")


