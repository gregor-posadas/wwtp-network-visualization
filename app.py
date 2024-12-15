# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

from io import StringIO

# -------------------------------
# Utility Functions
# -------------------------------

def remove_outliers_zscore(df, threshold=3):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        st.warning("No numeric columns found to apply Z-Score filtering.")
        return df
    z_scores = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    mask = (z_scores.abs() < threshold).all(axis=1)
    filtered_df = df[mask]
    st.write(f"Outliers removed: {len(df) - len(filtered_df)}")
    return filtered_df

def generate_heatmap(corr_matrix, title, container):
    with container:
        st.subheader(title)
        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            labels={"x": "Parameters", "y": "Parameters", "color": "Correlation Coefficient"},
            title=title,
        )
        fig.update_layout(
            autosize=True,
            width=1000,  # Adjust as needed
            height=800,  # Adjust as needed
            margin=dict(l=50, r=50, t=50, b=50)
        )
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Main Streamlit App
# -------------------------------

def main():
    st.set_page_config(page_title="WWTP Network Visualization", layout="wide")
    st.title("WWTP Unit Processes Network Visualization")
    st.write("""
    Welcome to the WWTP (Waste Water Treatment Plant) Unit Processes Network Visualization tool. 
    This application allows you to upload multiple data files, generate correlation heatmaps, and visualize the relationships between different processes and parameters within your WWTP system.
    """)

    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state['step'] = 'upload'
    if 'dataframes' not in st.session_state:
        st.session_state['dataframes'] = []
    if 'sorted_files' not in st.session_state:
        st.session_state['sorted_files'] = []
    if 'process_labels' not in st.session_state:
        st.session_state['process_labels'] = []

    if st.session_state['step'] == 'upload':
        upload_files()
    elif st.session_state['step'] == 'visualize':
        generate_visualizations()
    elif st.session_state['step'] == 'globally_shared_network':
        generate_globally_shared_network()
    elif st.session_state['step'] == 'locally_shared_network':
        generate_locally_shared_network()
    elif st.session_state['step'] == 'targeted_network':
        generate_targeted_network()

def upload_files():
    uploaded_files = st.file_uploader(
        "Upload CSV or Excel Files",
        accept_multiple_files=True,
        type=['csv', 'xlsx', 'xls']
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} files uploaded successfully.")

        st.subheader("Arrange Uploaded Files in Desired Order")
        file_order = pd.DataFrame({
            'Filename': [file.name for file in uploaded_files]
        })
        order_numbers = []
        total_files = len(uploaded_files)

        for idx, row in file_order.iterrows():
            number = st.number_input(
                f"Position for '{row['Filename']}'",
                min_value=1,
                max_value=total_files,
                value=idx+1,
                step=1,
                key=f"order_{idx}"
            )
            order_numbers.append(number)

        file_order['Order'] = order_numbers

        # Check for unique order assignments
        if file_order['Order'].nunique() != total_files:
            st.error("Each file must have a unique position number. Please adjust the positions to avoid duplicates.")
        else:
            sorted_filenames = file_order.sort_values('Order')['Filename']
            sorted_files = [file for filename in sorted_filenames for file in uploaded_files if file.name == filename]
            st.session_state['sorted_files'] = sorted_files

            # Prompt for process labels
            st.subheader("Assign Labels to Each Process")
            process_labels = []
            for idx, file in enumerate(sorted_files):
                label = st.text_input(
                    f"Label for '{file.name}':",
                    value=f"Process {idx + 1}",
                    key=f"label_{idx}"
                )
                process_labels.append(label)
            st.session_state['process_labels'] = process_labels

            if st.button("Confirm File Upload and Order"):
                # Process and store dataframes
                for file in sorted_files:
                    try:
                        if file.name.endswith(('.xlsx', '.xls')):
                            df = pd.read_excel(file)
                        else:
                            df = pd.read_csv(file)
                        df.columns = df.columns.str.lower().str.strip()
                        if 'date' not in df.columns:
                            st.error(f"The file **{file.name}** does not contain a 'date' column.")
                            return
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                        df = df.dropna(subset=["date"])
                        df = remove_outliers_zscore(df)
                        st.session_state['dataframes'].append(df)
                    except Exception as e:
                        st.error(f"Error processing file **{file.name}**: {e}")
                        return

                st.session_state['step'] = 'visualize'
                st.success("Files processed successfully! Proceed to generate visualizations.")

def generate_visualizations():
    st.subheader("Generate Visualizations")

    # Generate heatmaps
    st.write("### Correlation Heatmaps")
    for idx in range(len(st.session_state['dataframes']) - 1):
        process_label_1 = st.session_state['process_labels'][idx]
        process_label_2 = st.session_state['process_labels'][idx + 1]
        merged_df = pd.merge(
            st.session_state['dataframes'][idx],
            st.session_state['dataframes'][idx + 1],
            on="date",
            suffixes=(f"_{idx}", f"_{idx + 1}")
        )
        merged_df = merged_df.drop(columns=["date"], errors="ignore").select_dtypes(include=[float, int])

        if not merged_df.empty:
            corr_matrix = merged_df.corr()
            title = f"Correlation Coefficient Heatmap: {process_label_1} vs {process_label_2}"
            generate_heatmap(corr_matrix, title, st.container())
        else:
            st.warning(f"No numeric data available for Heatmap {idx + 1}.")

    st.write("---")

    # Buttons for network diagrams
    st.subheader("Generate Network Diagrams")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Generate Globally Shared Network Diagram"):
            st.session_state['step'] = 'globally_shared_network'

    with col2:
        if st.button("Generate Locally Shared Network Diagram"):
            st.session_state['step'] = 'locally_shared_network'

    st.write("---")

    # Buttons for additional visualizations
    col3, col4 = st.columns(2)

    with col3:
        if st.button("Generate Bar Chart for Globally Shared Parameters"):
            # Placeholder for bar chart generation
            st.session_state['step'] = 'bar_chart'

    with col4:
        if st.button("Generate Line Graph for Globally Shared Parameters"):
            # Placeholder for line graph generation
            st.session_state['step'] = 'line_graph'

    st.write("---")

    # Button for Targeted Network Diagram
    if st.button("Generate Targeted Network Diagram"):
        st.session_state['step'] = 'targeted_network'

def generate_globally_shared_network():
    st.subheader("Globally Shared Network Diagram")
    st.write("**This feature is under development.**")
    # Implement your globally shared network diagram logic here
    # Ensure to handle session_state appropriately to maintain app state

def generate_locally_shared_network():
    st.subheader("Locally Shared Network Diagram")
    st.write("**This feature is under development.**")
    # Implement your locally shared network diagram logic here
    # Ensure to handle session_state appropriately to maintain app state

def generate_bar_chart():
    st.subheader("Bar Chart: Globally Shared Parameter Correlations")
    st.write("**This feature is under development.**")
    # Implement your bar chart generation logic here

def generate_line_graph():
    st.subheader("Line Graph: Globally Shared Parameter Correlations")
    st.write("**This feature is under development.**")
    # Implement your line graph generation logic here

def generate_targeted_network():
    st.subheader("Targeted Network Diagram")
    st.write("**This feature is under development.**")
    # Implement your targeted network diagram logic here
    # Ensure to handle session_state appropriately to maintain app state

if __name__ == "__main__":
    main()