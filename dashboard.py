import pandas as pd
import streamlit as st
import numpy as np
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import geopandas
import plotly.express as px
from datetime import datetime, time


st.set_page_config(layout='wide')
@st.cache(allow_output_mutation=True)


# Functions
def get_data(path):
    # Read Path
    data = pd.read_csv(path)

    # Delete Duplicates and drop non used columns
    data = data.drop_duplicates(subset='id')
    data = data.drop(columns=['sqft_living15', 'sqft_lot15'], axis=1)
    # Convert data types
    data['date'] = pd.to_datetime( data['date'] ).dt.strftime( '%Y-%m-%d' )
    # data['lat'] = data['lat'].astype(str)
    # data['long'] = data['long'].astype(str)
    #data['zipcode'] = data['zipcode'].astype(str)

    # Create a query columns using 'lat' + 'long'
    # data['lat,long'] = data[['lat','long']].apply(lambda x: x['lat'] +','+x['long'],axis = 1)
    return data

@st.cache(allow_output_mutation=True)
def get_geofile(url):
    geofile = geopandas.read_file(url)
    return geofile

# Add New Features
def add_price_m2(data):
    # 1 sqft = 0.092903 m²
    conversion_constant = 0.092903
    data['price_m2'] = round(data['price'] / data['sqft_lot'] * conversion_constant, 2)
    return data


def overview_data(data):
    # Data Overview
    st.sidebar.title('Overview Data')
    f_attributes = st.sidebar.multiselect('Enter Columns', data.columns)
    f_zipcode = st.sidebar.multiselect('Zipcode', data['zipcode'].sort_values().unique())

    # Filtering data using  atributes and zipcodes
    if (f_attributes != []) & (f_zipcode != []):  # attributes and zipcode selected
        data = data.loc[data['zipcode'].isin(f_zipcode), f_attributes]
    elif (f_attributes == []) & (f_zipcode != []):  # attributes no selected
        data = data.loc[data['zipcode'].isin(f_zipcode), :]
    elif (f_attributes != []) & (f_zipcode == []):  # zipcode non selected
        data = data.loc[:, f_attributes]
    else:  # attributes and zipcode non selected
        data = data.copy()

    ##########################################
    # Average Metrics
    ##########################################
    df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df3 = data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    df4 = data[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

    # Merge datasets
    n1 = pd.merge(df1, df2, on='zipcode', how='inner')
    n2 = pd.merge(df3, df4, on='zipcode', how='inner')
    df = pd.merge(n1, n2, on='zipcode', how='inner')
    df.columns = ['ZIPCODE', 'TOTAL HOUSES', 'PRICE ($)', 'SQFT LIVING', 'PRICE ($) / M2']

    ##########################################
    # Statistic Desciptive
    ##########################################
    num_attributes = data.select_dtypes(include=['int64', 'float64'])
    mean = pd.DataFrame(num_attributes.apply(np.mean))
    median = pd.DataFrame(num_attributes.apply(np.median))
    std = pd.DataFrame(num_attributes.apply(np.std))
    min_ = pd.DataFrame(num_attributes.apply(np.min))
    max_ = pd.DataFrame(num_attributes.apply(np.max))
    df1 = pd.concat([max_, min_, mean, median, std], axis=1).reset_index()
    df1.columns = ['ATTRIBUTES', 'MAX', 'MIN', 'MEAN', 'MEDIAN', 'STD']

    # Load -> Showing Data
    st.title(':house: House Rocket :rocket: - Data Overview ')
    st.dataframe(data)

    c1, c2 = st.beta_columns((1, 1))
    # Average Metrics Table
    c1.header(':signal_strength: Average Values')
    c1.dataframe(df, height=600)

    # Statistic Desciptive Table
    c2.header(':chart_with_upwards_trend: Statistic Descriptive')
    c2.dataframe(df1, height=600)

    return None

def portfolio_density(data,geofile):
    st.title('Region Overview')

    c1, c2 = st.beta_columns((2, 2), gap='large')

    df = data.sample(10)

    # Folium Lib - Base Map
    c1.header('Portfolio Density')
    density_map = folium.Map(location=[data['lat'].mean(),
                                       data['long'].mean()],
                             default_zoom_start=15)

    marker_cluster = MarkerCluster().add_to(density_map)
    for name, row in df.iterrows():
        folium.Marker([row['lat'], row['long']],
                      popup='Sold R$ {0} on {1}. Features: {2} sqft, '
                            '{3} bedrooms, {4} bathrooms, year_built: {5}'.format(row['price'],
                                                                                  row['date'],
                                                                                  row['sqft_living'],
                                                                                  row['bedrooms'],
                                                                                  row['bathrooms'],
                                                                                  row['yr_built'])
                      ).add_to(marker_cluster)
    with c1:
        folium_static(density_map)

    # Region Price Map
    c2.header('Price Density')
    df = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df.columns = ['ZIP', 'PRICE']

    df = df.sample(10)
    geofile = geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]

    region_price_map = folium.Map(location=[data['lat'].mean(),
                                            data['long'].mean()],
                                  default_zoom_start=15)

    folium.Choropleth(data=df,
                      geo_data=geofile,
                      columns=['ZIP', 'PRICE'],
                      key_on='feature.properties.ZIP',
                      fill_color="YlOrRd",
                      nan_fill_color="White",
                      fill_opacity=0.8,
                      line_opacity=0.4,
                      legend_name='AVG PRICE'
                      ).add_to(region_price_map)

    with c2:
        folium_static(region_price_map)

def commercial(data):
    st.sidebar.title('Commercial Options')
    st.title('Commercial Attributes')

    # ========= Average Price per Year =========
    st.header('Average Price per Year Built')

    # Filters
    st.sidebar.subheader('Select Max Year Built')
    f_year_built = st.sidebar.slider('Year Built',
                                     int(data['yr_built'].min()),
                                     int(data['yr_built'].max()),
                                     int(data['yr_built'].max()))

    # Select Data
    df = data.loc[data['yr_built'] < f_year_built]
    df = df[['yr_built', 'price']].groupby(by=['yr_built']).mean().reset_index()

    # Ploting
    fig = px.line(df, x='yr_built', y='price')
    fig.add_hline(y=df.price.mean(), line_dash="dot", line_color='red', )
    st.plotly_chart(fig, use_container_width=True)

    # ========= Average Price per Day =========
    st.header('Average Price per Day Offered for Sale')

    # Filters
    st.sidebar.subheader('Select Max Date')

    # load data
    data = get_data(path='kc_house_data.csv')
    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')

    # setup filters
    min_date = datetime.strptime(data['date'].min(), '%Y-%m-%d')
    max_date = datetime.strptime(data['date'].max(), '%Y-%m-%d')
    f_date = st.sidebar.slider('Date', min_date, max_date, max_date)

    data['date'] = pd.to_datetime(data.date)
    df = data.loc[data['date'] < f_date]

    df = df[['date', 'price']].groupby(by=['date']).mean().reset_index()

    fig = px.line(df, x='date', y='price')
    fig.update_traces(line_color='green')
    fig.add_hline(y=df.price.mean(), line_dash="dot", line_color='red')
    st.plotly_chart(fig, use_container_width=True)

    # ========= Average Price per Day =========
    st.header('Price Histogram')

    # Filters
    st.sidebar.subheader('Select Max Price')
    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    avg_price = int(data['price'].mean())
    f_price = st.sidebar.slider('Select Max Price',
                                min_price,
                                max_price,
                                avg_price)

    df = data.loc[data['price'] < f_price]

    fig = px.histogram(df, x='price', nbins=50)
    st.plotly_chart(fig, use_container_width=True)
    return None


def physical_categories(data):
    st.title('Attributes Options')

    # Add Feature
    data['is_waterfront'] = data['waterfront'].apply(lambda x: 'No' if x == 0 else 'Yes')

    # Filters
    st.sidebar.subheader('House Attributes')

    f_bedrooms = st.sidebar.selectbox('Houses per Bedrooms',
                                      data['bedrooms'].sort_values().unique(),
                                      index=len(data['bedrooms'].sort_values().unique()) - 1)

    f_bathrooms = st.sidebar.selectbox('Houses per Bathrooms',
                                       data['bathrooms'].sort_values().unique(),
                                       index=len(data['bathrooms'].sort_values().unique()) - 1)

    f_floors = st.sidebar.selectbox('Max number of Floors',
                                    data['floors'].sort_values().unique(),
                                    index=len(data['floors'].sort_values().unique()) - 1)

    f_waterfront = st.sidebar.selectbox('Only Houses with Water View',
                                        data['is_waterfront'].sort_values().unique(),
                                        index=1)

    c1, c2 = st.beta_columns(2)

    # ========= House per Bedrooms ==========
    c1.header('House per Bedrooms')
    df = data[data['bedrooms'] <= f_bedrooms]
    fig = px.histogram(df, x='bedrooms', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    # ========= House per Bathrooms =========
    c2.header('House per Bathrooms')
    df = data[data['bathrooms'] <= f_bathrooms]
    fig = px.histogram(df, x='bathrooms', nbins=10)
    c2.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.beta_columns(2)
    # ========= House per Floors ============
    c1.header('House per Floors')
    df = data[data['floors'] < f_floors]
    fig = px.histogram(df, x='floors', nbins=10)
    c1.plotly_chart(fig, use_container_width=True)

    # ========= House per Water View ========
    c2.header('Houses with Water View')
    if f_waterfront == 'Yes':
        df = data[data['is_waterfront'] == 'Yes']
    fig = px.histogram(df, x='is_waterfront', nbins=2)
    c2.plotly_chart(fig, use_container_width=True)
    return None


if __name__== '__main__':
    # data extratction
    path = 'kc_house_data.csv'
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'

    data = get_data(path)
    geofile = get_geofile(url)

    # Transformation
    data = add_price_m2(data)

    # Graphics and Tables
    overview_data(data)
    portfolio_density(data, geofile)
    commercial(data)
    physical_categories(data)

