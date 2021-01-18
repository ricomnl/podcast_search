import os
import datetime
import faiss
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from vector_engine.utils import vector_search
import urllib.request

os.environ["TOKENIZERS_PARALLELISM"] = "false"

@st.cache(allow_output_mutation=True)
def read_data(data="data/lex_fridman_all_sentences_processed.parquet"):
    """Read the data from S3."""
    url = "https://podcast-search-scify.s3.amazonaws.com/lex_fridman_all_sentences_processed.parquet"
    return pd.read_parquet(url).to_dict("records")


@st.cache(allow_output_mutation=True)
def load_bert_model(name="distilbert-base-nli-stsb-mean-tokens"):
    """Instantiate a sentence-level DistilBERT model."""
    return SentenceTransformer(name)


@st.cache(allow_output_mutation=True)
def load_faiss_index(path_to_faiss="models/lex_similar_sentences.index"):
    """Load and deserialize the Faiss index."""

    data = urllib.request.urlopen("https://podcast-search-scify.s3.amazonaws.com/lex_similar_sentences_distil.index")
    reader = faiss.PyCallbackIOReader(data.read)
    index = faiss.read_index(reader)

    return index


def timestamp_to_seconds(timestamp):
  """ Takes a timestamps string like 00:00:00 and returns the amount of seconds passed. """
  date_time = datetime.datetime.strptime(timestamp, "%H:%M:%S")
  a_timedelta = date_time - datetime.datetime(1900, 1, 1)
  return int(a_timedelta.total_seconds())


def main():
    # Load data and models
    data = read_data()

    # index_selected = st.sidebar.multiselect('Select index', ["information_retrieval", "similar_sentences", "similar_questions"], "similar_sentences")
    # model_selected = {
    #     "information_retrieval": "distilroberta-base-msmarco-v2",
    #     "similar_sentences": "roberta-large-nli-stsb-mean-tokens",
    #     "similar_questions": "distilbert-base-nli-stsb-quora-ranking",
    # }
    # model_name = model_selected[index_selected[0]]
    # index_name = index_selected[0]


    model_name = "distilbert-base-nli-stsb-mean-tokens"
    index_name = "similar_sentences"

    faiss_index = load_faiss_index()
    model = load_bert_model(model_name)

    st.title("Semantic Podcast Search Demo")
    st.write("Search across all of the transcripts of Lex Fridman's AI Podcast. The underlying model will find the most similar sentences to your input.")

    # User search
    user_input = st.text_area("Search box", "what are the limits of deep learning?")

    # Filters
    st.sidebar.markdown("**Filters**")

#     filter_year = st.sidebar.slider("Publication year", 2010, 2021, (2010, 2021), 1)
#     filter_citations = st.sidebar.slider("Citations", 0, 250, 0)
    num_results = st.sidebar.slider("Number of search results", 10, 500, 50)
    titles = pd.DataFrame(data)["title"].unique()
    titles_selected = st.sidebar.multiselect('Filter by episode', titles)

    st.sidebar.text("")
    st.sidebar.write("Made by rico - [website](https://ricomeinl.com) - [twitter](https://twitter.com/rmeinl)")

    # Fetch results
    if user_input:
        # Get paper IDs
        if user_input == "?":
            indices = np.array([i for (i, obj) in enumerate(data) if "?" in obj["sentence"]])
        else:
            distance, indices = vector_search([user_input], model, faiss_index, num_results)

        # Get individual results
        for id_ in indices.flatten().tolist():
            obj = (data[id_])

            if len(titles_selected) == 0 or obj["title"] in set(titles_selected):
                st.write(obj["title"])
                st.write(f'https://youtu.be/{obj["videoId"]}?t={timestamp_to_seconds(obj["time"])-15}')
                st.write(obj["sentence"])
                st.write("------------------------------------------------------------")
                st.write()


if __name__ == "__main__":
    main()
