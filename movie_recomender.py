import pandas as pd
from embedding_function import MovieEmbedderFunction
from datetime import datetime
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import Ollama



def create_chroma_collections(dataframe, name):
    db = Chroma(
      name=name, embedding_function=MovieEmbedderFunction(), persist_directory="./movies.chroma")
    if len(db) == 0:
         ids = []
         metadatas = []
         documents = []
         for index, row in dataframe.iterrows():
            movie_id = row['movieId']
            title = row['title']
            genres = row['genres']
            avg_rating = row['avg_rating']
            year = row['year']
            metadata = {
                'genres': genres,
                'avg_rating': avg_rating,
                'year': year
            }
            documents.append(title)
            metadatas.append(metadata)
            ids.append(str(movie_id))

         db.add_texts(documents, metadatas, ids)

    return db


# data processing
moviesDataFrame = pd.read_csv('./dataset/movies.csv')
ratings = pd.read_csv('./dataset/ratings.csv')
# links = pd.read_csv('./dataset/links.csv')
tags = pd.read_csv('./dataset/tags.csv')

# make a copy of the original titles
movie_title_series = moviesDataFrame['title']


# remove the year from the title
moviesDataFrame['title'] = movie_title_series.apply(
    lambda title: title.split('(')[0].strip() if '(' in title else title)
# add a new column in the data frame to store the year
moviesDataFrame['year'] = movie_title_series.apply(
    lambda year: year.split('(')[-1].strip() if '(' in year else year)
moviesDataFrame['year'] = moviesDataFrame['year'].str.replace(')', '')

# compute the avg_rating for each movie
avg_ratings = ratings.groupby(
    ['movieId'])['rating'].mean().rename('avg_rating')
# merge the avg_rating series into the movie data frame
moviesDataFrame = moviesDataFrame.merge(avg_ratings, on='movieId', how='left')

# group the tags by movie_id
movie_tags = tags.groupby('movieId')['tag'].agg(
    lambda x: list(set(x))).rename('tags')


# merge the movie_tags series into the movie data frame
moviesDataFrame = moviesDataFrame.merge(movie_tags, on='movieId', how='left')

s = datetime.now()
movie_title_collection = create_chroma_collections(
    moviesDataFrame, 'movie_titles_collection')

print(f"Chroma db created in  in {(datetime.now() - s).total_seconds()} seconds")


# Setup LLM
llm = Ollama(model="mistral:latest")


# Your system + user instructions combined in one prompt template with {context} and {question}
template = """
You are a strict movie information assistant.
You can only answer using the text provided in the Reference Passage below.
If the answer is not in the passage, say exactly: "I could not find relevant information in the reference data."
Do not guess. Do not use any external knowledge.

Reference Passage:
{context}

QUESTION: {question}
"""

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)


# Setup RetrievalQA with custom prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=movie_title_collection.as_retriever(search_type="similarity", search_kwargs={"k":5}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=False,
)


st.title("Movie Bot")

# React to user input
if prompt := st.chat_input("Give me a movie"):
    st.chat_message("user").markdown(prompt)
    response = qa_chain.run(prompt)
    with st.chat_message("assistant"):
        st.markdown(f"Movie Bot: {response}")

