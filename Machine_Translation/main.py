import streamlit as st
import numpy as np
import nltk
import pdb
import pickle
import string
from os import getcwd
from utils import cosine_similarity


def nearest_neighbor(v, candidates, k=1, cosine_similarity=cosine_similarity):
    """
    Input:
      - v, the vector you are going find the nearest neighbor for
      - candidates: a set of vectors where we will find the neighbors
      - k: top k nearest neighbors to find
    Output:
      - k_idx: the indices of the top k closest vectors in sorted form
    """
    similarity_l = []

    # for each candidate vector...
    for row in candidates:
        # get the cosine similarity
        cos_similarity = cosine_similarity(v, row)

        # append the similarity to the list
        similarity_l.append(cos_similarity)

    # sort the similarity list and get the indices of the sorted list
    sorted_ids = np.argsort(np.array(similarity_l))

    # Reverse the order of the sorted_ids array
    sorted_ids = np.flip(sorted_ids)

    # get the indices of the k most similar candidate vectors
    k_idx = sorted_ids[:k]
    return k_idx

# add folder, tmp2, from our local workspace containing pre-downloaded corpora files to nltk's data path
filePath = f"{getcwd()}/tmp2/"
nltk.data.path.append(filePath)

# Load english and french word embedding
en_embeddings_subset = pickle.load(open("./data/en_embeddings.p", "rb"))
fr_embeddings_subset = pickle.load(open("./data/fr_embeddings.p", "rb"))

st.title("Machine Translation English to French")
st.write("The full dataset for English embeddings is about 3.64 gigabytes, and the French embeddings are about 629 megabytes. To safety for memory, in this project I will use subset of the embeddings for the words that we'll use in this assignment. So with some word in English you may not translate it")

with open("R.npy", "rb") as f:
    R_train = np.load(f)

with open("Y.npy", "rb") as f:
    Y_subset = np.load(f)

english_word = st.text_input("Your english word: ")

if english_word in en_embeddings_subset:
    english_word_embedding = en_embeddings_subset[english_word]
    french_word_embedding = np.dot(english_word_embedding, R_train)
    idx = nearest_neighbor(french_word_embedding, Y_subset, 1)
    for i in fr_embeddings_subset.keys():
        if (np.array_equal(fr_embeddings_subset[i], Y_subset[idx.item()])):
            st.write("French word is: ", i)
else:
    st.write("Can't translate")