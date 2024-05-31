import streamlit as st
import pandas as pd
from PIL import Image
import os

csv_file = 'output.csv' 
data = pd.read_csv(csv_file)

original_data = data.copy()

data['non_nan_count'] = data.notna().sum(axis=1)

sorted_data = data.sort_values(by='non_nan_count', ascending=False)

sorted_data = sorted_data.drop(columns=['non_nan_count'])

st.title('Vehicle Information Dashboard')

st.subheader('Data Table')
st.dataframe(original_data)

st.subheader('Records with Images')

def display_record(row):
    file_name = row['file_name']
    image_path = os.path.join('data', file_name)
    
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption=file_name, use_column_width=True)
    else:
        st.text(f'Image not found: {file_name}')
    
    st.write('**Registration Number:**', row['registration number'])
    st.write('**Registration Date:**', row['registration date'])
    st.write('**Manufacturing Date:**', row['manufacturing date'])
    st.write('**Chassis Number:**', row['chassis number'])
    st.write('**Engine Number:**', row['engine number'])
    st.write('**Name:**', row['name'])

for index, row in sorted_data.iterrows():
    st.write('---')
    display_record(row)
