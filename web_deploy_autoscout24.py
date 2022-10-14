#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 1 07:03:58 2022

@author: groupelegant
"""

#load the model from disk
import joblib
import numpy as np
import pandas as pd
import sklearn
#Import libraries
# from matplotlib.backend_bases import LocationEvent
import streamlit as st
from PIL import Image

filename = 'RandomForestRegressor01.sav'
model = joblib.load(filename)

#Import python scripts
from preprocessing_Elegant import (Gearboxd, body_typed, colourd, drivetraind,
                                   fuel_countryd, fuel_typed, locationd,
                                   make_modeld, n_dictd, preprocess,
                                   upholsteryd)


def main():
    #Setting Application title
    st.title('Group Elegant Model App')

      #Setting Application description
    st.markdown("""
     :dart:  WELCOME! This Streamlit app is made to predict AutoScout use case.
    The application is functional for online prediction. \n
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

       
        
    image = Image.open('App.png')
    image1 = Image.open('importance.png')
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict the Price of Autos')
    st.sidebar.image(image)
    st.sidebar.info('This app uses Random Forest Model')
    st.sidebar.image(image1)



    if add_selectbox == "Online":
        st.info("Input data below")
        st.subheader("The Most Important Features of Model:")
        #Based on our optimal features selection
        location = st.selectbox('Location of the Auto' , (locationd))
        First_Registration_Year = st.slider('First Registration Year' ,1980,2022,2020)     #st.number_input('First Registration Year of the Car', min_value=1940, max_value=2022, value=2000)
        Power_kW = st.slider('Power kW', 0,220,100)
        Empty_Weight_kg = st.slider('Empty Weight (kg)', 300,5000,2000)
        mileage = st.slider('Mileage of Vehicle (km) ', 1000,300000,100000)
        gears = st.selectbox('Gear of the Car', ('1','2','3', '4','5','6','7', '8','9'))
        make_model = st.selectbox('Make and Model of the Auto' , (make_modeld))
        cylinders = st.selectbox('Cylinder of the Car', ('1','2','3', '4','5','6','7', '8','9','10'))
        Engine_Size_cc = st.slider('Engine Size (cc)', 600,3000,1400)
        Gearbox = st.selectbox('Gearbox', (Gearboxd))
        fuel_type = st.selectbox('Fuel Type', (fuel_typed))
        fuel_country = st.selectbox('Fuel Country', (fuel_countryd))
        fuel_city = st.slider('Fuel City', 0,16,10)
        fuel_comb = st.slider('Fuel Comb', 0,10,5)
        co2_emissions = st.slider('CO2 Emission per km', 0,300,90)
        body_type = st.selectbox('Body Type', (body_typed))
        seats = st.selectbox('Seat of the Car', ('1','2','3', '4','5','6','7', '8','9'))
        doors = st.selectbox('Door of the Car', ('1','2','3', '4','5','6','7', '8','9'))
        colour = st.selectbox('Colour', (colourd))
        upholstery = st.selectbox('Upholstery', (upholsteryd))
        drivetrain = st.selectbox('Drivetrain', (drivetraind))

        
        

        
        data = {
            'mileage': mileage,
            'fuel_type': fuel_typed[fuel_type],
            'seller': 0,
            'body_type': body_typed[body_type],
            'type': 0,
            'drivetrain': drivetraind[drivetrain],
            'seats': n_dictd[seats],
            'doors': n_dictd[doors],
            'Warranty_Months': 0,
            'full_service_history': 0,
            'non_smoker_vehicle': 0,
            'Gearbox': Gearboxd[Gearbox],
            'Engine_Size_cc': Engine_Size_cc,
            'gears': n_dictd[gears],
            'cylinders': n_dictd[cylinders],
            'Empty_Weight_kg': Empty_Weight_kg,
            'co2_emissions': co2_emissions,
            'Emission_Class_Euro': 0,
            'colour': colourd[colour],
            'Paint': 0,
            'Upholstery_Color': 0,
            'upholstery': upholsteryd[upholstery],
            'location': locationd[location],
            'fuel_city': fuel_city,
            'fuel_country': fuel_countryd[fuel_country],
            'fuel_comb': fuel_comb,
            'Power_kW': Power_kW,
            'First_Registration_Year': First_Registration_Year,
            '2_zones': 0,
            '360°_camera': 0,
            '3_zones': 0,
            '4_zones': 0,
            'Air_conditioning': 0,
            'Air_suspension': 0,
            'Armrest': 0,
            'Automatic_climate_control': 0,
            'Auxiliary_heating': 0,
            'Cruise_control': 0,
            'Electric_backseat_adjustment': 0,
            'Electric_tailgate': 0,
            'Electrical_side_mirrors': 0,
            'Electrically_adjustable_seats': 0,
            'Electrically_heated_windshield': 0,
            'Fold_flat_passenger_seat': 0,
            'Heads_up_display': 0,
            'Heated_steering_wheel': 0,
            'Hill_Holder': 0,
            'Keyless_central_door_lock': 0,
            'Leather_seats': 0,
            'Leather_steering_wheel': 0,
            'Light_sensor': 0,
            'Lumbar_support': 0,
            'Massage_seats': 0,
            'Multi_function_steering_wheel': 0,
            'Navigation_system': 0,
            'Panorama_roof': 0,
            'Park_Distance_Control': 0,
            'Parking_assist_system_camera': 0,
            'Parking_assist_system_self_steering': 0,
            'Parking_assist_system_sensors_front': 0,
            'Parking_assist_system_sensors_rear': 0,
            'Power_windows': 0,
            'Rain_sensor': 0,
            'Seat_heating': 0,
            'Seat_ventilation': 0,
            'Sliding_door_left': 0,
            'Sliding_door_right': 0,
            'Split_rear_seats': 0,
            'Start_stop_system': 0,
            'Sunroof': 0,
            'Tinted_windows': 0,
            'Wind_deflector': 0,
            'ABS': 0,
            'Adaptive_Cruise_Control': 0,
            'Adaptive_headlights': 0,
            'Alarm_system': 0,
            'Bi_Xenon_headlights': 0,
            'Blind_spot_monitor': 0,
            'Central_door_lock': 0,
            'Central_door_lock_with_remote_control': 0,
            'Daytime_running_lights': 0,
            'Distance_warning_system': 0,
            'Driver_drowsiness_detection': 0,
            'Driver_side_airbag': 0,
            'Electronic_stability_control': 0,
            'Emergency_brake_assistant': 0,
            'Emergency_system': 0,
            'Fog_lights': 0,
            'Full_LED_headlights': 0,
            'Glare_free_high_beam_headlights': 0,
            'Head_airbag': 0,
            'High_beam_assist': 0,
            'Immobilizer': 0,
            'Isofix': 0,
            'LED_Daytime_Running_Lights': 0,
            'LED_Headlights': 0,
            'Lane_departure_warning_system': 0,
            'Laser_headlights': 0,
            'Night_view_assist': 0,
            'Passenger_side_airbag': 0,
            'Power_steering': 0,
            'Rear_airbag': 0,
            'Side_airbag': 0,
            'Speed_limit_control_system': 0,
            'Tire_pressure_monitoring_system': 0,
            'Traction_control': 0,
            'Traffic_sign_recognition': 0,
            'Xenon_headlights': 0,
            'Android_Auto': 0,
            'Apple_CarPlay': 0,
            'Bluetooth': 0,
            'CD_player': 0,
            'Digital_cockpit': 0,
            'Digital_radio': 0,
            'Hands_free_equipment': 0,
            'Induction_charging_for_smartphones': 0,
            'Integrated_music_streaming': 0,
            'MP3': 0,
            'On_board_computer': 0,
            'Radio': 0,
            'Sound_system': 0,
            'Television': 0,
            'USB': 0,
            'WLAN_/_WiFi_hotspot': 0,
            'make_model': make_modeld[make_model]


                }
        # print(data.values())
        features_df = pd.DataFrame.from_dict([data])
        
        print(features_df)

        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        
        st.dataframe(features_df)

    


        #Preprocess inputs
        # preprocess_df = preprocess(features_df, 'Online')

        # features_df_scaled = sc.transform(features_df)

        prediction = model.predict(features_df)

        if st.button('Predict'):
            if prediction == 1:
                #st.warning(f'Your vehicles value is € {int(prediction)}')
                st.warning(prediction)
        

    # else:
    #     st.subheader("Dataset upload")
        # uploaded_file = st.file_uploader("Choose a file")
        # if uploaded_file is not None:
        #     data = pd.read_csv(uploaded_file,encoding= 'utf-8')
        #     #Get overview of data
        #     st.write(data.head())
        #     st.markdown("<h3></h3>", unsafe_allow_html=True)
        #     #Preprocess inputs
        #     preprocess_df = preprocess(data, "Batch")
        #     if st.button('Predict'):
        #        #Get batch prediction
        #         prediction = model.predict(preprocess_df)
        #         prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
        #         prediction_df = prediction_df.replace({1:'Yes, the passenger survive.', 0:'No, the passenger died'})

        #         st.markdown("<h3></h3>", unsafe_allow_html=True)
        #         st.subheader('Prediction')
        #         st.write(prediction_df)
            
if __name__ == '__main__':
        main()
