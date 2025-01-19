from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

model = load_model('Final Lightgbm Model Jan2025')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['prediction_label'][0]
    return predictions

def run():
    image = Image.open('JALA_Logo.png')

    st.image(image, use_container_width=False)

    add_selectbox = st.sidebar.selectbox(
    "Ingin melakukan prediksi Survival Rate via apa?",
    ("Online", "CSV"))

    st.sidebar.info('Aplikasi ini digunakan untuk memprediksi Survival Rate budidaya Udang')
    

    st.title("Survival Rate Prediction App")

    if add_selectbox == 'Online':

        # Kolom Kategori Density
        total_seed = st.number_input('Total Bibit (Seed)', min_value=1, max_value=1000000, value=100)
        area = st.number_input('Luas Area / Ukuran Kolam (m²)', min_value=1, max_value=1000000, value=50)
        density = total_seed/area
        density_category = True

        if density <= 80:
            density_category =  True
        elif 80 < density <= 200:
            density_category =  False
        else:
            density_category =  False

        # Kolom total_seed_type
        total_seed_type = True
        input_total_seed_type = st.selectbox('Total Jenis Bibit (Seed Type)', ['actual', 'net', 'gross'])

        if input_total_seed_type == 'actual':
            total_seed_type = False
        elif input_total_seed_type == 'gross':
            total_seed_type = True
        else:
            total_seed_type = False

        # Kolom exceeds_3/4_months
        exceeds_3_4_months = True
        input_exceeds_3_4_months = st.selectbox('Apakah Lama Budidaya Melebihi 3-4 Bulan', ['Ya', 'Tidak'])

        if input_exceeds_3_4_months == 'Ya' :
            exceeds_3_4_months =  True
        else:
            exceeds_3_4_months = False
        
        # Kolom initial_age
        initial_age = st.number_input('Usia Budidaya Saat Ini (Hari)', min_value=1, max_value=100, value=10)

        # Kolom weight_per_seed
        weight_per_seed = 0
        weight = st.number_input('Berapa Berat Timbangan Udang tearkhir (pastikan memiliki estimasi berat atau hasil timbangan terakhir)', min_value=1, max_value=10000, value=25)
        weight_per_seed = weight/total_seed

        # Kolom total_seed_per_area
        total_seed_per_area = total_seed/area

        # Kolom target_size
        target_size = st.number_input('Target Total Jumlah udang per-Kg', min_value=1, max_value=100, value=10)

        # Predict
        output=""

        input_dict = {'density_category_Rendah (≤80 PL/m²)' : density_category, 
                      'total_seed_type_gross' : total_seed_type, 
                      'weight_per_seed' : weight_per_seed, 
                      'initial_age' : initial_age, 
                      'exceeds_3/4_months_no' : exceeds_3_4_months,
                      'target_size' : target_size,
                      'total_seed_per_area' : total_seed_per_area}
        
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = float(output)
            output = str(round(output,2)) + '%'

        st.success('Hasil Survival Rate nya adalah {}'.format(output))

    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload CSV file for predictions", type=["csv"])
        
        if file_upload is not None:
            try:
                data = pd.read_csv(file_upload)
                st.write("Data yang diupload:")
                st.write(data.head())  
                predictions = predict_model(estimator=model, data=data)
                st.write("Hasil prediksi untuk data yang diupload:")
                st.write(predictions)  
                
            except Exception as e:
                st.error(f"Error: {e}") 
        else:
            st.info("Upload File Kamu!") 
            
if __name__ == '__main__':
    run()