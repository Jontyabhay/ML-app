import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

def main():
    st.title('COVID-19 spread impact web app')
    st.sidebar.title('COVID-19 user classifier')
    st.sidebar.markdown('Is your country under great impact?')

    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv("covid_19_data.csv")
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col].astype(str))
        return data

    df = load_data()

    if st.sidebar.checkbox("Show raw data",False):
        st.subheader("Covid dataset")
        st.write(df)        




if __name__ == '__main__':
    main()    