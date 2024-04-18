import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from function import *
from streamlit_option_menu import *


url = "https://github.com/bayuprnmji31/DATA-MINING_1/raw/main/social%20media%20influencers%20-%20Youtube%20sep-2022.csv"
df = pd.read_csv(url)


url2 = "https://raw.githubusercontent.com/bayuprnmji31/DATA-MINING_1/main/Data%20Final.csv"
df2 = pd.read_csv(url2)

   
# Membuat tampilan dashboard
with st.sidebar :
    selected = option_menu('Analisis Pengaruh YouTube dalam Strategi Pemasaran',['Introducing','Data Distribution','Relation','Composition & Comparison','Regression'],default_index=0)

if (selected == 'Introducing'):
    st.title('Analisis Pengaruh YouTube dalam Strategi Pemasaran')
    country_counts = df['Country'].value_counts()
    max_count_index = country_counts.idxmax()

    fig_geo = go.Figure(data=go.Choropleth(
        locations=country_counts.index,
        z=country_counts.values,
        locationmode='country names',
        colorscale='Blues',
        marker_line_color='black',
        colorbar_title='Jumlah Youtuber'
    ))

    fig_geo.update_layout(
        title='Analisis Geografis Youtuber',
        geo=dict(
            showcoastlines=True,
            projection_type='equirectangular'
        )
    )

    st.plotly_chart(fig_geo)
    st.write("""
        YouTube telah menjadi salah satu platform pemasaran yang sangat penting dalam era digital saat ini. 
        Dengan jutaan pengguna yang aktif setiap hari, YouTube menawarkan potensi besar untuk mencapai audiens target Anda 
        dan memengaruhi keputusan pembelian mereka.
    """)
    st.image('youtube1.jpg', caption='Analisis Pengaruh YouTube dalam Strategi Pemasaran', use_column_width=True)



if selected == 'Data Distribution':
    st.header("Distribusi Data")
    scatter_plot(df)  
    
if (selected == 'Relation'):
    st.title('Hubungan')
    heatmap(df)

if selected == 'Composition & Comparison':
    st.title('Komposisi & Perbandingan')
    numerical_feature = 'Avg Likes'
    categorical_feature = 'Country'
        
    df_60 = df.head(60)
    
    composition_and_comparison(df_60, numerical_feature, categorical_feature)


if (selected == 'Regression'):
    st.title('Regresi Random Foresty!')
    # random_forest_model = regression(df)
    regresi()