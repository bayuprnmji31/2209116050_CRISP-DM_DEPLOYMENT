import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import pickle
import joblib

def load_data():
    url = "https://raw.githubusercontent.com/bayuprnmji31/DATA-MINING_1/main/social%20media%20influencers%20-%20Youtube%20sep-2022.csv"
    df = pd.read_csv(url)
    return df
# Load the data
data = load_data()
st.write(data)

file_path_clf = 'random_forest_model.pkl'
with open(file_path_clf, 'rb') as f:
    clf = joblib.load(f)
    

def scatter_plot(df):
    st.subheader("Distribusi Data untuk Analisis Pengaruh YouTube dalam Strategi Pemasaran")

    # Convert data to proper format for Plotly Express
    df['Avg Views'] = df['Avg. views\r\n'].replace({',': '', 'K': 'e3', 'M': 'e6'}, regex=True).astype(float)
    df['Avg Likes'] = df['Avg. likes'].replace({',': '', 'K': 'e3', 'M': 'e6'}, regex=True).astype(float)
    df['Avg Comments'] = df['Avg Comments'].replace({',': '', 'K': 'e3', 'M': 'e6'}, regex=True).astype(float)

    # Plot scatter plot using Plotly Express
    fig = px.scatter(df, x='Avg Views', y='Avg Likes', size='Avg Comments', 
                     title='Pengaruh Avg Views dan Avg Likes terhadap Avg Comments',
                     labels={'Avg Views': 'Rata-rata Ditonton', 'Avg Likes': 'Rata-rata Suka', 'Avg Comments': 'Rata-rata Komentar'},
                     color_continuous_scale='blues')

    fig.update_layout(title_font_size=20, xaxis_title_font_size=16, yaxis_title_font_size=16)
    fig.update_traces(marker=dict(size=10, opacity=0.7))
    st.plotly_chart(fig)

    # Plot geographical analysis
    country_counts = df['Country'].value_counts()
    max_count_index = country_counts.idxmax()

    fig_geo = go.Figure(data=go.Choropleth(
        locations=country_counts.index,
        z=country_counts.values,
        locationmode='country names',
        colorscale='Reds',
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


def convert_to_numeric(value):
    try:
        if 'K' in value:
            return float(value.replace('K', '')) * 1e3
        elif 'M' in value:
            return float(value.replace('M', '')) * 1e6
        else:
            return float(value)
    except ValueError:
        return None

def heatmap(df):
    st.subheader("Heatmap of Numeric Features Correlation")

    # Convert columns to numeric
    df['Avg Views'] = df['Avg. views\r\n'].replace({',': '', 'K': 'e3', 'M': 'e6'}, regex=True).astype(float)
    df['Avg Likes'] = df['Avg. likes'].replace({',': '', 'K': 'e3', 'M': 'e6'}, regex=True).astype(float)
    df['Avg Comments'] = df['Avg Comments'].replace({',': '', 'K': 'e3', 'M': 'e6'}, regex=True).astype(float)

    # Select numerical features and drop non-numeric ones
    numerical_df = df.select_dtypes(include=['float64', 'int64'])

    # Compute correlation matrix
    corr_matrix = numerical_df.corr()

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Heatmap of Numeric Features Correlation")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    st.write("""
        Penjelasan Gambar Heatmap of Numeric Features Correlation
        
        Gambar tersebut menunjukkan heatmap korelasi antar fitur numerik. Heatmap adalah representasi visual dari data matriks, di mana setiap sel mewakili nilai korelasi antara dua variabel. Warna sel menunjukkan kekuatan dan arah korelasi.
        
        Elemen-elemen gambar:

        S.no: Angka seri dari fitur numerik.
        Judul kolom: Nama fitur numerik.
        Nilai korelasi: Nilai korelasi antara dua fitur numerik, ditunjukkan dengan warna.
        Skala warna: Skala warna yang digunakan untuk menunjukkan kekuatan dan arah korelasi.
        Penjelasan nilai korelasi:

        Nilai korelasi positif: Menunjukkan hubungan positif antara dua variabel. Semakin tinggi nilai korelasi, semakin kuat hubungannya.
        Nilai korelasi negatif: Menunjukkan hubungan negatif antara dua variabel. Semakin tinggi nilai korelasi, semakin kuat hubungannya.
        Nilai korelasi 0: Menunjukkan tidak ada hubungan antara dua variabel.
        Contoh interpretasi:

        Korelasi antara "Avg Comments" dan "Avg Views" adalah 0.58. Ini menunjukkan bahwa terdapat hubungan positif yang sedang antara dua variabel tersebut. Artinya, semakin banyak komentar yang diterima postingan, semakin banyak pula penayangan yang didapatnya.
        Korelasi antara "Avg Likes" dan "Avg Views" adalah 0.73. Ini menunjukkan bahwa terdapat hubungan positif yang kuat antara dua variabel tersebut. Artinya, semakin banyak likes yang diterima postingan, semakin banyak pula penayangan yang didapatnya.
        Korelasi antara "Avg Likes" dan "Avg Comments" adalah 0.71. Ini menunjukkan bahwa terdapat hubungan positif yang kuat antara dua variabel tersebut. Artinya, semakin banyak likes yang diterima postingan, semakin banyak pula komentar yang didapatnya.
            """)


def composition_and_comparison(df, numerical_feature=None, categorical_feature=None):
    if numerical_feature in df:
        # Histogram
        st.subheader(f"Distribusi dari {numerical_feature}")
        fig_hist = px.histogram(df, x=numerical_feature, title=f"Distribusi {numerical_feature}")
        st.plotly_chart(fig_hist)

        # Box Plot
        st.subheader(f"Box Plot dari {numerical_feature} berdasarkan {categorical_feature}")
        fig_box = px.box(df, x=categorical_feature, y=numerical_feature, title=f"Box Plot {numerical_feature} berdasarkan {categorical_feature}")
        st.plotly_chart(fig_box)

        # Violin Plot
        st.subheader(f"Violin Plot dari {numerical_feature} berdasarkan {categorical_feature}")
        fig_violin = px.violin(df, x=categorical_feature, y=numerical_feature, title=f"Violin Plot {numerical_feature} berdasarkan {categorical_feature}")
        st.plotly_chart(fig_violin)

    if categorical_feature and categorical_feature in df.columns:
        # Bar Plot
        st.subheader(f"Analisis Komposisi: Proporsi {categorical_feature}")
        fig_bar = px.bar(df[categorical_feature].value_counts(), 
                         x=df[categorical_feature].value_counts().index, 
                         y=df[categorical_feature].value_counts().values,
                         labels={'x': categorical_feature, 'y': 'Jumlah'})
        fig_bar.update_layout(title=f"Analisis Komposisi: {categorical_feature}", xaxis_title=categorical_feature, yaxis_title="Jumlah")
        st.plotly_chart(fig_bar)
        
        # Pie Chart
        st.subheader(f"Pie Chart: Proporsi {categorical_feature}")
        fig_pie = px.pie(df, names=categorical_feature, title=f"Pie Chart: Proporsi {categorical_feature}")
        st.plotly_chart(fig_pie)
    elif categorical_feature:
        st.write(f"Error: '{categorical_feature}' bukan kolom yang valid dalam DataFrame.")



def load_data2():
    url = "https://raw.githubusercontent.com/bayuprnmji31/DATA-MINING_1/main/Data%20Final.csv"
    df2 = pd.read_csv(url)
    return df2

def predict_regression(avg_views, avg_likes, avg_comments, clf):
    data = load_data2()

    X = data[['Avg Views', 'Avg. likes', 'Avg Commentss']]
    y = data[' Subscribers']  

    clf.fit(X, y)

    prediction = clf.predict([[avg_views, avg_likes, avg_comments]])
    return prediction[0]

def regression_prediction_input():
    st.subheader("Prediksi Regresi Jumlah Pelanggan YouTube")
    st.write("Masukkan data untuk memprediksi jumlah pelanggan YouTube.")

    avg_views = st.number_input("Rata-rata Ditonton", min_value=0)
    avg_likes = st.number_input("Rata-rata Suka", min_value=0)
    avg_comments = st.number_input("Rata-rata Komentar", min_value=0)

    return avg_views, avg_likes, avg_comments

def regresi():
    st.title('Aplikasi Prediksi Jumlah Pelanggan YouTube')
    st.write('Gunakan aplikasi ini untuk memprediksi jumlah pelanggan YouTube.')

    # Load the model
    file_path_clf = 'random_forest_model.pkl'
    with open(file_path_clf, 'rb') as f:
        clf = joblib.load(f)

    # Input for regression prediction
    avg_views, avg_likes, avg_comments = regression_prediction_input()

    # Predict button
    if st.button('Prediksi'):
        subscribers_prediction = predict_regression(avg_views, avg_likes, avg_comments, clf)
        st.write(f'Prediksi Jumlah Pelanggan: {subscribers_prediction:.2f}')

