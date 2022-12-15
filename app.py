import streamlit as st

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

import nltk

#Visualization libraries
from plotly.offline import init_notebook_mode, iplot 
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as py

# Disable warnings
import warnings
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image

warnings.filterwarnings('ignore')

st.markdown("<h1 style='text-align: center; color: white; font-size: 60px;'>Youtube Content Creation Recommendation Platform</h1>", unsafe_allow_html=True)
code=['Brazil','Canada','France','UK','India','Japan','Canada','US','None']
country_= st.radio("Choose country",code, index=7)

if country_ =='Brazil':
    country='BR'
elif country_ == 'Canada':
    country='CA'
elif country_ == 'France':
    country='FR'
elif country_ == 'UK':
    country='GB'
elif country_ == 'India':
    country='IN'
elif country_ == 'Japan':
    country='JP'
elif country_ == 'Canada':
    country='CA'
elif country_ == 'US':
    country='US'
else:
    st.write('Choose a country')
    st.title("")
    st.title("")
    st.title("")
    st.title("")
    st.title("")
    st.title("")
    st.title("")
    st.title("")
    st.title("")
    st.title("")
    st.title("")
    st.title("")
    st.title("")
    


data=pd.read_csv(""+country+"_youtube_trending_data.csv")
st.header('Statistics of Dataframe')
st.write(data.describe())
st.header('Header of Dataframe')
st.write(data.head())
st.header('Shape of Dataframe')
st.text(data.shape)
st.header('Null values:')
st.write(data.isna().sum())
st.header('Shape after dropping the null values:')
data=data.dropna()
st.text(data.shape)
st.header("Count of unique values in the Dataframe")
for i in data.columns:
    x=i+":",len(data[str(i)].value_counts())
    st.text(x)
    st.text("-------------------------------------")
st.markdown("**We can see repeated video IDs. Hence we will consdiered the video_id with most recent date as it contians cummulative comment and like count.**")
data["video_id"].value_counts()[:20]
data['trending_date']= pd. to_datetime(data['trending_date'])
data["month"]=pd. DatetimeIndex(data["trending_date"]).month
data["day"]=pd. DatetimeIndex(data["trending_date"]).day
data["week"]=pd. DatetimeIndex(data["trending_date"]).week
data=data.drop(columns=["thumbnail_link","channelId"],axis=1)
fdata=data.sort_values(by="trending_date").drop_duplicates(subset=["video_id"], keep="last")
st.markdown("## Shape after keeping latest dated video id")
st.text(fdata.shape)
data=fdata.sort_values(by=['view_count'],ascending=False)

fdata=fdata.sort_values(by=['view_count'],ascending=False)

st.title('View count by channel title')
top20views = fdata[:20]
fig = px.bar(top20views, x='channelTitle', y='view_count',color='view_count', hover_data=['view_count',"title"])
fig.update_xaxes(title_text='Channel title',title_font = {"size": 14},tickfont=dict(family='Rockwell', size=10))
fig.update_yaxes(title_text='View_count in Millions',title_font = {"size": 14},tickfont=dict(family='Rockwell', size=10))
st.write(fig)

st.title('Average View count by Category id')
cat_count=fdata.groupby("categoryId")["view_count"].mean()
fig = px.bar(cat_count, x=cat_count.index, y=cat_count.values ,color=cat_count.values)
fig.update_xaxes(title_text='Category Id',title_font = {"size": 14},tickfont=dict(family='Rockwell', size=10))
fig.update_yaxes(title_text='Average View Count',title_font = {"size": 14},tickfont=dict(family='Rockwell', size=10))
st.write(fig)

st.title('Average View count by Weekdays')
fdata['publishedAt']= pd. to_datetime(fdata['publishedAt'])
fdata["Published_day"]=pd. DatetimeIndex(fdata["publishedAt"]).day
fdata["Published_week"]=pd. DatetimeIndex(fdata["publishedAt"]).week
fdata["Published_time"]=pd. DatetimeIndex(fdata["publishedAt"]).time
fdata["Published_weekday"]=pd. DatetimeIndex(fdata["publishedAt"]).weekday
day_count=fdata.groupby("Published_weekday")["view_count"].mean()
color_code=["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
fig = px.scatter(day_count, x=day_count.index, y=day_count.values, color=color_code,color_discrete_sequence=px.colors.qualitative.Vivid,size=day_count.values*10)
fig.update_xaxes(title_text='Weekdays',title_font = {"size": 14},tickfont=dict(family='Rockwell', size=10))
fig.update_yaxes(title_text='Average View Count',title_font = {"size": 14},tickfont=dict(family='Rockwell', size=10))
st.write(fig)

stop_words = set(stopwords.words('english')) 
if country=='BR':
    stop_words = set(stopwords.words('spanish'))
elif country=='FR':
    stop_words = set(stopwords.words('french')) 
fdata['title'] = fdata.title.apply(lambda x: word_tokenize(x))
fdata['title'] = fdata.title.apply(lambda x: [w for w in x if w not in stop_words])
fdata['title'] = fdata.title.apply(lambda x: ' '.join(x))

fdata['tags'] = fdata.tags.apply(lambda x: word_tokenize(x))
fdata['tags'] = fdata.tags.apply(lambda x: [w for w in x if w not in stop_words])
fdata['tags'] = fdata.tags.apply(lambda x: ' '.join(x))

fdata['description'] = fdata.description.apply(lambda x: word_tokenize(x))
fdata['description'] = fdata.description.apply(lambda x: [w for w in x if w not in stop_words])
fdata['description'] = fdata.description.apply(lambda x: ' '.join(x))

pop_words=[]
def WordCloudfunction(title,text):
    cloudtext=' '.join(fdata[text].tolist())
    sns.set(rc={'figure.figsize':(16,10)})
    wordcloud = WordCloud().generate(cloudtext)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.title(title,size=24)
    st.image(wordcloud.to_array())
    text_dictionary = wordcloud.process_text(cloudtext)
    # sort the dictionary
    word_freq={k: v for k, v in sorted(text_dictionary.items(),reverse=True, key=lambda item: item[1])}
    #use words_ to print relative word frequencies
    rel_freq=wordcloud.words_
    #print results
    pop=list(word_freq.items())[:5]
    pop_words.extend(pop)
    st.markdown("Most popular:")
    st.text(pop)
    
st.title('Most popular tags')
WordCloudfunction('Tags','tags')
st.title('Most popular Video Titles')
WordCloudfunction('Video title','title')
st.title('Most popular Descriptions')
WordCloudfunction('Description','description')
st.title('Most popular Channel Titles')
WordCloudfunction('channelTitle','channelTitle')
