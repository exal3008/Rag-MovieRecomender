import pandas as pd
from datetime import datetime
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers import EnsembleRetriever
import re


MAX_BATCH = 5461

@st.cache_resource
def load_movies_dataset():
    # data processing
    moviesDataFrame = pd.read_csv('./dataset/movies.csv')
    ratings = pd.read_csv('./dataset/ratings.csv')
    # links = pd.read_csv('./dataset/links.csv')
    tags = pd.read_csv('./dataset/tags.csv')

    # make a copy of the original titles
    movie_title_series = moviesDataFrame['title']

    # remove the year from the title
    moviesDataFrame['title'] = moviesDataFrame['title'] = movie_title_series.apply(
        lambda t: re.sub(r'\(\d{4}\)\s*$', '', t).strip()
    )
    # add a new column in the data frame to store the year

    # Robust extraction: find all 4-digit numbers in parentheses and pick the last one
    # list of all 4-digit numbers per title
    years = movie_title_series.str.findall(r'\((\d{4})\)')
    # pick the last match, or None
    years = years.apply(lambda x: x[-1] if x else None)
    moviesDataFrame['year'] = pd.to_numeric(
        years, errors='coerce').astype('Int64')

    # compute the avg_rating for each movie
    avg_ratings = ratings.groupby(
        ['movieId'])['rating'].mean().rename('avg_rating')
    # merge the avg_rating series into the movie data frame
    moviesDataFrame = moviesDataFrame.merge(
        avg_ratings, on='movieId', how='left')

    # group the tags by movie_id
    movie_tags = tags.groupby('movieId')['tag'].agg(
        lambda x: '|'.join(sorted(set(str(x))))).rename('tags')

    # merge the movie_tags series into the movie data frame
    moviesDataFrame = moviesDataFrame.merge(
        movie_tags, on='movieId', how='left')

    return moviesDataFrame


@st.cache_resource
def create_chroma_collections(name):
    embedding = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(
        collection_name=name, embedding_function=embedding, persist_directory="./movies.chroma")
    print('Length  ids' + str(len(db.get()['ids'])))
    if len(db.get()['ids']) == 0:
        dataframe = load_movies_dataset()
        ids = []
        metadatas = []
        documents = []
        for index, row in dataframe.iterrows():
            movie_id = row['movieId']
            title = row['title']
            genres = row['genres']
            avg_rating = row['avg_rating']
            tags = row['tags']
            year_value = row['year']
            if pd.isna(year_value):   # check for <NA>
                year_value = None
            metadata = {
                'genres': genres,
                'avg_rating': avg_rating,
                'year': year_value,
                'tags': tags
            }
            documents.append(title)
            metadatas.append(metadata)
            ids.append(str(movie_id))
            if (len(documents) == MAX_BATCH):
                db.add_texts(ids=ids,
                             texts=documents,
                             metadatas=metadatas)
                documents = []
                ids = []
                metadatas = []
         # Add leftover data after loop
            if documents:
             db.add_texts(ids=ids, texts=documents, metadatas=metadatas)

    return db


s = datetime.now()
movie_title_collection = create_chroma_collections('movie_titles_collection')

print(
    f"Chroma db created in  in {(datetime.now() - s).total_seconds()} seconds")


metadata_field_info = [
    AttributeInfo(
        name="genres",
        description="Genres of the movie Can be one or multiple, separated by |",
        type="string or list of strings",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="avg_rating",
        description="The rating for the movie",
        type="float"
    ),
    AttributeInfo(
        name='tags',
        description="Aditional info about the movie",
        type="string or list of strings"
    )
]


# Setup LLM
llm = ChatOllama(model="mistral:latest")

self_query_retriever = SelfQueryRetriever.from_llm(
    llm,
    movie_title_collection,
    'Brief summary of a movie',
    metadata_field_info,
    search_kwargs={"k": 10},  # fetch 20 documents instead of 5
    exclude_keys=["title"]  # ignore title filtering
)


ensemble_retriever = EnsembleRetriever(
    retrievers=[self_query_retriever,
                movie_title_collection.as_retriever(search_kwargs={"k": 10})],
    weights=[0.3, 0.7]  # adjust depending on importance
)



st.title("Movie Bot")



# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



if prompt := st.chat_input("Give me a movie"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

     #  Display the user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    progress_text = st.empty()
    progress_bar = st.progress(0)

    with st.spinner("Movie Bot is thinking..."):
        results = ensemble_retriever.invoke(prompt)
        context = ""
        total_docs = len(results)

        for i, doc in enumerate(results):
            context += (
                f"Title: {doc.page_content}\n"
                f"Genres: {doc.metadata.get('genres', 'N/A')}\n"
                f"Year: {doc.metadata.get('year', 'N/A')}\n"
                f"Avg Rating: {doc.metadata.get('avg_rating', 'N/A')}\n\n"
                f"Tags: {doc.metadata.get('tags', 'N/A')}\n\n"
            )

            progress_bar.progress(int((i+1)/total_docs*50))
            progress_text.text(f"Processing document {i+1}/{total_docs}...")

        # System + user prompt
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

        prompt_text = custom_prompt.format(context=context, question=prompt)

        # Simulate LLM progress
        for i in range(51, 101):
            progress_bar.progress(i)
            progress_text.text(f"Querying Movie Bot... {i}%")
        response = llm.invoke(prompt_text)

    # Clear progress UI
    progress_text.empty()
    progress_bar.empty()

   
    st.session_state.messages.append({"role": "assistant", "content": response.text()})
    with st.chat_message("assistant"):
        st.markdown(response.text())