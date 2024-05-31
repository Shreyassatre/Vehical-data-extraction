import streamlit as st
import pandas as pd
from PIL import Image
import os

# Load the CSV file
csv_file = 'output.csv'  # Replace with your actual CSV file path
data = pd.read_csv(csv_file)

# Make a copy of the original data to display the table in its original order
original_data = data.copy()

# Count the number of non-NaN values for each record
data['non_nan_count'] = data.notna().sum(axis=1)

# Sort the data based on the non-NaN count in descending order for detailed display
sorted_data = data.sort_values(by='non_nan_count', ascending=False)

# Drop the non_nan_count column as it's no longer needed for display
sorted_data = sorted_data.drop(columns=['non_nan_count'])

# Set up the Streamlit app
st.title('Vehicle Information Dashboard')

# Display the data in the table in its original order
st.subheader('Data Table')
st.dataframe(original_data)

# Display each record with its corresponding image, sorted by non-NaN count
st.subheader('Records with Images')

# Define a function to display records
def display_record(row):
    file_name = row['file_name']
    image_path = os.path.join('data', file_name)  # Image directory
    
    
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

# Iterate through the sorted dataframe and display each record
for index, row in sorted_data.iterrows():
    st.write('---')
    display_record(row)
