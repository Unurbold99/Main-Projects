import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit App Main title
st.title('Extensive EDA Web App')

# Code that enables fo the user to upload their CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    st.subheader("Data Overview")

    df = pd.read_csv(uploaded_file)

    # Display basic data information
    st.write(f"**Number of Rows:** {df.shape[0]}")
    st.write(f"**Number of Columns:** {df.shape[1]}")
    st.write("")

    st.subheader("Data Summary")

    # Display summary statistics
    st.write(df.describe())

    st.subheader("Data Preview")

    # Display a sample of the data
    st.write(df.head())

    st.subheader("Data Columns")

    # Display the list of columns
    st.write(df.columns)

    st.subheader("Missing Values")

    # Display a table with missing values
    st.write(df.isnull().sum())

    st.subheader("Data Types")

    # Display the data types of columns
    st.write(df.dtypes)

    st.subheader("Correlation Matrix")

    # Display a correlation matrix
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    st.pyplot()

    st.subheader("Numeric Features Distributions")

    # Display histograms for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number])
    for col in numeric_cols.columns:
        st.write(f"**{col}**")
        fig, ax = plt.subplots()
        ax.hist(numeric_cols[col])
        st.pyplot(fig)

    st.subheader("Categorical Features Counts")

    # Display bar plots for categorical columns
    categorical_cols = df.select_dtypes(include=['object'])
    for col in categorical_cols.columns:
        st.write(f"**{col}**")
        sns.countplot(data=df, x=col)
        plt.xticks(rotation=45)
        st.pyplot()