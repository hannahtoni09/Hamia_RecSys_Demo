import pickle
import streamlit as st
import requests
import pandas as pd
import numpy as np

# set page setting
st.set_page_config(page_title='iSpeak Courses')

# set history var
if 'history' not in st.session_state:
    st.session_state.history = []

# import preprocessed data
data = pd.read_csv("courses.csv")
data['tags'] = data['name'] + " - " + data['description']
data = data.rename(columns = {'name':'title'})
#-----------------------------------------------

# # import similarity (to be cached)
# def importSim(filename):
#     sim = pickle.load(open(filename, 'rb'))
#     return sim

# similarity = importSim('C:/MachineLearning/Hamia_Recommendation_System/Hamia_RecSys/similarity.pkl')
# ----------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
tfid = TfidfVectorizer()
vectorTfid =tfid.fit_transform(data['tags']).toarray()
similarityTfidVect = cosine_similarity(vectorTfid)


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words = 'english', lowercase = True)
vector=cv.fit_transform(data['tags']).toarray()
similarityCountVect = cosine_similarity(vector)


from sklearn.feature_extraction.text import HashingVectorizer
hash = HashingVectorizer(lowercase = True, ngram_range = (1,1))
vectorHash = hash.fit_transform(data['tags']).toarray()
similarityHasingVect = cosine_similarity(vectorHash)

res = np.multiply(similarityTfidVect, similarityCountVect )
similarity = np.multiply(res, similarityHasingVect)

# ----------------------------------------------
# update last viewed list
def update_las_viewed():
    if len(st.session_state.history) > 3:
        st.session_state.history.pop()

# sidebar
st.sidebar.write("""
This is a content based recommender system. Pick a course from the list or search for it and then wait for the recommendations.
You will get six courses, posters and descriptions. 
Made by Hamia Research 
""")

# title
st.write("# iSpeak Course Recommendation System")
st.write("Pick a course from the list and enjoy some new stuffs!")

# select box
title = st.selectbox("", data["title"])
if title not in st.session_state.history:
    st.session_state.history.insert(0, title)
update_las_viewed()

#------------------------

# recommender function
def recommend_image(course, sim):
    poster = []
    plot = []
    description = []
    # index from dataframe
    index = data[data['title'] == course].index[0]
    dist = dict(enumerate(sim[index]))
    dist = dict(sorted(dist.items(), reverse=True, key = lambda item: item[1]))
    #index from 1 because the first is the movie itself
    cnt = 0
    for key in dist: 
        cnt = cnt+1
        if cnt < 15:
            title = data.iloc[key].title
            try:
                posterRes, plotRes, plotDes = get_poster_plot(title) 
                poster.append(posterRes)
                plot.append(plotRes)
                description.append(plotDes)
            except:
                pass
        else:
            break
        
    return poster[1:], plot[1:]

# get poster
def get_poster_plot(title):
    # r = requests.get("http://www.omdbapi.com/?i=tt3896198&apikey=37765f04&t=" + title).json()
    posterElement = 'https://cdn-www.vinid.net/09523dba-banner2-copy-1.jpg'
    plotElement = title
    data_des = data[data['title'] == title]
    desElement = data_des['description'].values[0]
    return posterElement, plotElement, desElement

# ------------------------

# recommend
with st.spinner("Getting the best movies..."):
    recs, plots = recommend_image(title, similarity)

# recommendation cols
st.write("## Students study this course also choose the following ....")
# col1 = st.columns(1)
# with col1:
st.image(recs[0])
st.subheader(plots[0])
# st.write(des[0])
col2, col3 = st.columns(2)
with col2:
    st.image(recs[1])
    st.write(plots[1])
    # st.write(des[1])
with col3:
    st.image(recs[2])
    st.write(plots[2])
    # st.write(des[2])

col4, col5 = st.columns(2)
with col4:
    st.image(recs[3])
    st.write(plots[3])
    # st.write(des[3])
with col5:
    st.image(recs[4])
    st.write(plots[4])
    # st.write(des[4])

col6, col7 = st.columns(2)
with col6:
    st.image(recs[5])
    st.write(plots[5])
    # st.write(des[5])
    

with col7:
    st.image(recs[6])
    st.write(plots[6])
    # st.write(des[6])
# with col8:
#     st.image(recs[7])
#     st.write(plots[7])
# with col9:
#     st.image([8])
#     st.write(plots[8])

# last viewed
st.write("## Last viewed:")
r1, r2, r3 = st.columns(3)
with r1:
    try:
        st.image(get_poster_plot(st.session_state.history[0])[0])
        st.write(get_poster_plot(st.session_state.history[0])[1])
    except IndexError:
        pass
    
with r2:
    try:
        st.image(get_poster_plot(st.session_state.history[1])[0])
        st.write(get_poster_plot(st.session_state.history[1])[1])
    except IndexError:
        pass
    
with r3:
    try:
        st.image(get_poster_plot(st.session_state.history[2])[0])
        st.write(get_poster_plot(st.session_state.history[2])[1])
    except IndexError:
        pass
    
