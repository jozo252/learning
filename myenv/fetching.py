# streamlit_app.py
from scrape import *
import streamlit as st
import numpy as np
import os
from io import StringIO
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf



def fetch(ticker,data_func,name):
    try:
        output_dir = r"C:\Users\pc\Documents\cestakodovanim\learning\myenv\Data"
        folder_name = ticker
        folder_path = os.path.join(output_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)  # Create folder if not exists

        # Fetch and process income statement data
        income_statement_data = data_sort(data_func(ticker))  # Ensure data_sort() returns a DataFrame
        file_path = os.path.join(folder_path, f'{name}.csv')

        # If file exists, compare the latest date before overwriting
        if os.path.exists(file_path):
            existing_file = pd.read_csv(file_path)
            existing_file['Date'] = pd.to_datetime(existing_file['Date'], errors='coerce')

            latest_new_date = income_statement_data['Date'].iloc[0]  # Get latest date from new data
            latest_existing_date = existing_file['Date'].iloc[0]  # Get latest date from existing file

            # Compare latest dates
            if latest_new_date != latest_existing_date:
                income_statement_data.to_csv(file_path, index=False)
                print(f"Updated file: {file_path}")
            else:
                print(f"No update needed: {file_path} is already up-to-date.")
        else:
            # If file does not exist, create it
            income_statement_data.to_csv(file_path, index=False)
            print(f"File created at: {file_path}")
    except:
        print('error in fetch')

fetch('nio',revenue,'revenue')


