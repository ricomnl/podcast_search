import datetime
import faiss
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from vector_engine.utils import vector_search


@st.cache(allow_output_mutation=True)
def read_data(data="data/lex_fridman_all_sentences_processed.parquet"):
    """Read the data from S3."""
    return pd.read_parquet(data).to_dict("records")


@st.cache(allow_output_mutation=True)
def load_bert_model(name="roberta-large-nli-stsb-mean-tokens"):
    """Instantiate a sentence-level DistilBERT model."""
    return SentenceTransformer(name)


@st.cache(allow_output_mutation=True)
def load_faiss_index(path_to_faiss="models/lex_similar_sentences.index"):
    """Load and deserialize the Faiss index."""
    return faiss.read_index(path_to_faiss)


def timestamp_to_seconds(timestamp):
  """ Takes a timestamps string like 00:00:00 and returns the amount of seconds passed. """
  date_time = datetime.datetime.strptime(timestamp, "%H:%M:%S")
  a_timedelta = date_time - datetime.datetime(1900, 1, 1)
  return int(a_timedelta.total_seconds())


def main():
    # Load data and models
    data = read_data()

    index_selected = st.sidebar.multiselect('Select index', ["information_retrieval", "similar_sentences", "similar_questions"], "similar_sentences")
    model_selected = {
        "information_retrieval": "distilroberta-base-msmarco-v2",
        "similar_sentences": "roberta-large-nli-stsb-mean-tokens",
        "similar_questions": "distilbert-base-nli-stsb-quora-ranking",
    }

    faiss_index = load_faiss_index(f"models/lex_{index_selected[0]}.index")
    model = load_bert_model(model_selected[index_selected[0]])

    st.title("Semantic Podcast Search Demo")

    # User search
    user_input = st.text_area("Search box", "what are the limits of deep learning?")

    # Filters
    st.sidebar.markdown("**Filters**")
#     filter_year = st.sidebar.slider("Publication year", 2010, 2021, (2010, 2021), 1)
#     filter_citations = st.sidebar.slider("Citations", 0, 250, 0)
    num_results = st.sidebar.slider("Number of search results", 10, 500, 50)
    titles = pd.DataFrame(data)["title"].unique()
    titles_selected = st.sidebar.multiselect('Select titles', titles)

    # Fetch results
    if user_input:
        # Get paper IDs
        if user_input == "?":
            indices = np.array([i for (i, obj) in enumerate(data) if "?" in obj["sentence"]])
        else:
            distance, indices = vector_search([user_input], model, faiss_index, 1000)
        # Slice data on year
#         frame = data[
#             (data.year >= filter_year[0])
#             & (data.year <= filter_year[1])
#             & (data.citations >= filter_citations)
#         ]
        # Get individual results
        for id_ in indices.flatten().tolist():
#             if id_ in set(frame.id):
#                 f = frame[(frame.id == id_)]
#             else:
#                 continue
#             print()
#             print(q["link"])
#             print(q["sentence"])
            obj = (data[id_])

            if len(titles_selected) == 0 or obj["title"] in set(titles_selected):
                st.write(obj["title"])
                st.write(f'https://youtu.be/{obj["videoId"]}?t={timestamp_to_seconds(obj["time"])-15}')
                st.write(obj["sentence"])
                st.write("------------------------------------------------------------")
                st.write()


if __name__ == "__main__":
    main()
