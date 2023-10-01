from turtle import colormode
from track import *
import tempfile
import cv2
import torch
import streamlit as st
import os
from re import A
import plotly.graph_objects as go 
import numpy as np
import pandas as pd

# https://app.cpcbccr.com/AQI_India/ to downlaod the excel file

import cv2
from vidgear.gears import CamGear
from pyfirmata import Arduino
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
# st.markdown( """ <style> .sidebar .sidebar-content { background-image: linear-gradient(#2e7bcf,#2e7bcf); color: black; } </style> """, unsafe_allow_html=True )
st.set_page_config(page_title='Filter AQI Data', page_icon="ag logo.jpg")



#################################################################
# Main function of the program
if __name__ == '__main__':
    st.echo("hi")
    st.markdown("""<style>body {color: blue;background-color: #111;}</style>""", unsafe_allow_html=True)
    ###########################################################################
    # AQI data part
    # read by default 1st sheet of an excel file
    dataframe1 = pd.read_excel('march 2.xlsx')  # change the file name to the downloaded file downloaded from the "https://app.cpcbccr.com/AQI_India/"
    col1, col2= st.columns([3,2])
    ################################################################################################
    # data garbage removal

    # dropping the unwanted columns

    dataframe1.drop([0,1,2],axis=0,inplace=True)
    # changing the column names
    dataframe1.rename(columns = {'Central Pollution Control Board':'Sr. No.'}, inplace = True)
    dataframe1.rename(columns = {'Unnamed: 1':'State'}, inplace = True)
    dataframe1.rename(columns = {'Unnamed: 2':'City'}, inplace = True)
    dataframe1.rename(columns = {'Unnamed: 3':'Station Name'}, inplace = True)
    dataframe1.rename(columns = {'Unnamed: 4':'Current_AQI_value'}, inplace = True)

    # resetting the index of the data frame
    dataframe_indi = dataframe1.reset_index()

    # droping the coulmn 0 of the dataframe
    dataframe_indi.pop(dataframe_indi.columns[0])
    df = dataframe_indi.fillna(method='ffill') # this is for filling the vacc

    #########################################################
    # saving the dataframe to csv file
    df.to_csv('current_csv.csv',index=False)

    ########################################################
    # filtering the dataframe in streamlit webapp


    df = pd.read_csv("current_csv.csv")  # read a CSV file inside the 'data" folder next to 'app.py'
    ########################################################################
    # container for data and graph

    #############################################
    st.title("Kindly Filter the AQI data to process")
    aqi_val = 0

    # filter function
    def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:

        modify = st.checkbox("Add filters")

        if not modify:
            return df

        df = df.copy()

        # Try to convert datetimes into a standard format (datetime, no timezone)
        for col in df.columns:
            if is_object_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass

            if is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.tz_localize(None)

        modification_container = st.container()

        with modification_container:
            to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
            for column in to_filter_columns:
                left, right = st.columns((1, 20))
                left.write("↳")
                # Treat columns with < 10 unique values as categorical
                if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                    # print(column)
                    user_cat_input = right.multiselect(
                        f"Values for {column}",
                        df[column].unique(),
                        default=list(df[column].unique()),
                    )
                    df = df[df[column].isin(user_cat_input)]
                elif is_numeric_dtype(df[column]):
                    _min = float(df[column].min())
                    _max = float(df[column].max())
                    step = (_max - _min) / 100
                    user_num_input = right.slider(
                        f"Values for {column}",
                        _min,
                        _max,
                        (_min, _max),
                        step=step,
                    )
                    df = df[df[column].between(*user_num_input)]
                elif is_datetime64_any_dtype(df[column]):
                    user_date_input = right.date_input(
                        f"Values for {column}",
                        value=(
                            df[column].min(),
                            df[column].max(),
                        ),
                    )
                    if len(user_date_input) == 2:
                        user_date_input = tuple(map(pd.to_datetime, user_date_input))
                        start_date, end_date = user_date_input
                        df = df.loc[df[column].between(start_date, end_date)]
                else:
                    user_text_input = right.text_input(
                        f"Substring or regex in {column}",
                    )
                    if user_text_input:
                        df = df[df[column].str.contains(user_text_input)]
        if (len(df.head())==1):
            global aqi_val
            aqi_val= int(df.iat[0,4])
        return df


    df = pd.read_csv(
        "current_csv.csv"
    )

    ########################################################
    # gauge graph for AQI values

    def aqi_graph(value):
    #   print(graph_color)
        color = ''
        condn = ''
        if (aqi_val>=0 and aqi_val<=50):
          color = 'Green'
          condn = 'Good'
        elif (aqi_val>=51 and aqi_val<=100):
          color = '#90EE90'
          condn = 'Satisfactory'
        elif (aqi_val>=101 and aqi_val<=200):
          color = 'yellow'
          condn = 'Moderate'

        elif (aqi_val>=201 and aqi_val<=300):
          color = 'orange'
          condn = 'Poor'
        elif (aqi_val>=301 and aqi_val<=400):
          condn = 'Very Poor'
          color = 'red'
        elif (aqi_val>=401 and aqi_val<=500):
          color = '#8B0000' # hex code for dark red
          condn = 'Severe'
        elif(aqi_val>500):
          color = '#8B0000' # hex code for dark red
          condn = 'Severe'
        else:
          color = 'green'
          condn = 'Good'
        fig = go.Figure(go.Indicator(
          value = value,
          mode = "gauge+number",
          domain = {'x': [0, 1], 'y': [0, 1]},
          title = {'text': "AQI is {}".format(condn), 'font': {'size': 26}},
          gauge = {
              'axis': {'range': [None, 500]},
              'bar': {'color': color},# this is the bar color
              'bgcolor': "white",
              'borderwidth': 2,
              'bordercolor': "gray",
              'threshold': {
                  'line': {'color': "red", 'width': 4},
                  'thickness': 0.75,
                  'value': 401}}))

        # fig.show()
        return fig

    # accessing the function gauge_graph 
    # df.style.set_properties(**{"background-color": "black", "color": "lawngreen"})
    st.dataframe(filter_dataframe(df).style.set_properties(**{"background-color": "black", "color": "lawngreen"}),width=800)
    with st.sidebar:

        st.plotly_chart(aqi_graph(aqi_val), use_container_width=True,sharing="streamlit", theme="streamlit")
        st.sidebar.markdown('---')


################################################################################################################
    # condition check
    if(aqi_val<101):
        st.markdown('<h3 style="color: lightgreen"> All conditions are normal </h3', unsafe_allow_html=True)
        st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    if (aqi_val>100):
        st.markdown('<h4 style="color: lightgreen">AQI checked✅ </h4>', unsafe_allow_html=True)
        st.markdown('---')
        st.title('Operating on videos')
        video_file_buffer = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mov', 'avi'])

        if video_file_buffer:
            st.sidebar.text('Input video')
            st.sidebar.video(video_file_buffer)
            # save video from streamlit into "videos" folder for future detect
            with open(os.path.join('videos', video_file_buffer.name), 'wb') as f:
                f.write(video_file_buffer.getbuffer())

        st.sidebar.markdown('---')
        st.sidebar.title('Settings')
        # custom class
        custom_class = st.sidebar.checkbox('Custom classes')
        assigned_class_id = [0, 1, 2, 3]
        names = ['car', 'motorcycle', 'truck', 'bus']

        if custom_class:
            assigned_class_id = []
            assigned_class = st.sidebar.multiselect('Select custom classes', list(names))
            for each in assigned_class:
                assigned_class_id.append(names.index(each))

        # st.write(assigned_class_id)

        # setting hyperparameter
        confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.5)
        line = st.sidebar.number_input('Line position', min_value=0.0, max_value=1.0, value=0.6, step=0.1)
        st.sidebar.markdown('---')


        status = st.empty()
        stframe = st.empty()
        if video_file_buffer is None:
            status.markdown('<font size= "4"> **Status:** Waiting for input </font>', unsafe_allow_html=True)
        else:
            status.markdown('<font size= "4"> **Status:** Ready </font>', unsafe_allow_html=True)

        car, bus, truck, motor = st.columns(4)
        with car:
            st.markdown('**Car**')
            car_text = st.markdown('__')
            print(car_text)


        with bus:
            st.markdown('**Bus**')
            bus_text = st.markdown('__')

        with truck:
            st.markdown('**Truck**')
            truck_text = st.markdown('__')

        with motor:
            st.markdown('**Motorcycle**')
            motor_text = st.markdown('__')

        fps, _,  _, _  = st.columns(4)
        with fps:
            st.markdown('**FPS**')
            fps_text = st.markdown('__')


        track_button = st.sidebar.button('START')
        reset_button = st.sidebar.button('RESET ID')
        if track_button:
            # reset ID and count from 0
            reset()
            opt = parse_opt()
            opt.conf_thres = confidence
            opt.source = f'videos/{video_file_buffer.name}'

            # status.markdown('<font size= "4"> **Status:** Running... </font>', unsafe_allow_html=True)
            with torch.no_grad():
                detect(opt, stframe, car_text, bus_text, truck_text, motor_text, line, fps_text, assigned_class_id)
            status.markdown('<font size= "4"> **Status:** Finished ! </font>', unsafe_allow_html=True)
            # end_noti = st.markdown('<center style="color: blue"> FINISH </center>',  unsafe_allow_html=True)

        if reset_button:
            reset()
            st.markdown('<h3 style="color: blue"> Reseted ID </h3>', unsafe_allow_html=True)
    
