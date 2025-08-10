# Rag-MovieRecomender

## **Overview**:
This is a GEN AI  project that uses RAG to recommend movies to a user based on the criteria specified by the user  

This is a revised version of my initial Kaggle submission for [Gen AI Intensive Course Capstone 2025Q1](https://www.kaggle.com/code/exal30/capgenai).  

It includes the following changes:  
- Uses the Mistral model as an RAG agent  
- Has a chat-like UI built with **Streamlit**  
- Uses `sentence-transformers/all-MiniLM-L6-v2` to embed the movie information

## **Prerequisites**:
In order to run this project, you need the following dependencies installed:

- Python 3.x  
- pandas  
- chromadb  
- langchain  
- sentence_transformers  
- streamlit  
- langchain-community  
- langchain-ollama  

Install Ollama and then run the command:

```
ollama run mistral
```
## Dataset Setup

1. After you clone or checkout the source code, **create a folder named `dataset`** in the project root.

2. Download the MovieLens 32M dataset from [https://files.grouplens.org/datasets/movielens/ml-32m.zip](https://files.grouplens.org/datasets/movielens/ml-32m.zip)

3. Extract the contents of the zip file **into the `dataset` folder**. The folder structure should look like:  
   your_project/  
├── dataset/  
│ ├── movies.csv  
│ ├── ratings.csv  
│ ├── tags.csv  
│ └── ...  
├── your_code_files.py  
└── ...  

