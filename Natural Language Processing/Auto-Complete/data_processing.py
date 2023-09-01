import math
import random
import numpy as np
import pandas as pd
import nltk
import re
import pickle
nltk.download('punkt')
nltk.data.path.append('.')

with open("./data/en_US.twitter.txt", "r", encoding="utf8") as f:
    data = f.read()
print("Data type:", type(data))
print("Number of letters:", len(data))
print("First 300 letters of the data")
print("-------")
print("-------")

print("Last 300 letters of the data")
print("-------")
print("-------")


def split_to_sentences(data):
    """
    Split data by linebreak "\n"

    Args:
        data: str

    Returns:
        A list of sentences
    """
    sentences = data.split("\n")
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if len(s) > 0]

    return sentences


def tokenize_sentences(sentences):
    """
    Tokenize sentences into tokens (words)

    Args:
        sentences: List of strings

    Returns:
        List of lists of tokens
    """

    # Initialize the list of lists of tokenized sentences
    tokenized_sentences = []

    # Go through each sentence
    for sentence in sentences:  # complete this line

        # Convert to lowercase letters
        sentence = sentence.lower()

        # Convert into a list of words
        tokenized = nltk.word_tokenize(sentence)

        # append the list of words to the list of lists
        tokenized_sentences.append(tokenized)

    return tokenized_sentences


def get_tokenized_data(data):
    """
    Make a list of tokenized sentences

    Args:
        data: String

    Returns:
        List of lists of tokens
    """
    # Get the sentences by splitting up the data
    sentences = split_to_sentences(data)

    # Get the list of lists of tokens by tokenizing the sentences
    tokenized_sentences = tokenize_sentences(sentences)

    return tokenized_sentences

tokenized_data = get_tokenized_data(data)
random.seed(87)
random.shuffle(tokenized_data)

train_size = int(len(tokenized_data) * 0.8)
train_data = tokenized_data[0:train_size]
test_data = tokenized_data[train_size:]


def count_words(tokenized_sentences):
    """
    Count the number of word appearence in the tokenized sentences

    Args:
        tokenized_sentences: List of lists of strings

    Returns:
        dict that maps word (str) to the frequency (int)
    """

    word_counts = {}

    # Loop through each sentence
    for sentence in tokenized_sentences:  # complete this line

        # Go through each token in the sentence
        for token in sentence:  # complete this line

            # If the token is not in the dictionary yet, set the count to 1
            if token not in word_counts:  # complete this line with the proper condition
                word_counts[token] = 1

            # If the token is already in the dictionary, increment the count by 1
            else:
                word_counts[token] += 1

    return word_counts


def get_words_with_nplus_frequency(tokenized_sentences, count_threshold):
    """
    Find the words that appear N times or more

    Args:
        tokenized_sentences: List of lists of sentences
        count_threshold: minimum number of occurrences for a word to be in the closed vocabulary.

    Returns:
        List of words that appear N times or more
    """
    # Initialize an empty list to contain the words that
    # appear at least 'minimum_freq' times.
    closed_vocab = []

    # Get the word couts of the tokenized sentences
    # Use the function that you defined earlier to count the words
    word_counts = count_words(tokenized_sentences)

    #   filtered_words = None

    # for each word and its count
    for word, cnt in word_counts.items():  # complete this line

        # check that the word's count
        # is at least as great as the minimum count
        if cnt >= count_threshold:  # complete this line with the proper condition

            # append the word to the list
            closed_vocab.append(word)

    return closed_vocab


def replace_oov_words_by_unk(tokenized_sentences, vocabulary, unknown_token="<unk>"):
    """
    Replace words not in the given vocabulary with '<unk>' token.

    Args:
        tokenized_sentences: List of lists of strings
        vocabulary: List of strings that we will use
        unknown_token: A string representing unknown (out-of-vocabulary) words

    Returns:
        List of lists of strings, with words not in the vocabulary replaced
    """

    # Place vocabulary into a set for faster search
    vocabulary = set(vocabulary)

    # Initialize a list that will hold the sentences
    # after less frequent words are replaced by the unknown token
    replaced_tokenized_sentences = []

    # Go through each sentence
    for sentence in tokenized_sentences:

        # Initialize the list that will contain
        # a single sentence with "unknown_token" replacements
        replaced_sentence = []

        # for each token in the sentence
        for token in sentence:  # complete this line

            # Check if the token is in the closed vocabulary
            if token in vocabulary:  # complete this line with the proper condition
                # If so, append the word to the replaced_sentence
                replaced_sentence.append(token)
            else:
                # otherwise, append the unknown token instead
                replaced_sentence.append(unknown_token)

        # Append the list of tokens to the list of lists
        replaced_tokenized_sentences.append(replaced_sentence)
    return replaced_tokenized_sentences


def preprocess_data(train_data, test_data, count_threshold, unknown_token="<unk>",
                    get_words_with_nplus_frequency=get_words_with_nplus_frequency,
                    replace_oov_words_by_unk=replace_oov_words_by_unk):
    """
    Preprocess data, i.e.,
        - Find tokens that appear at least N times in the training data.
        - Replace tokens that appear less than N times by "<unk>" both for training and test data.
    Args:
        train_data, test_data: List of lists of strings.
        count_threshold: Words whose count is less than this are
                      treated as unknown.

    Returns:
        Tuple of
        - training data with low frequent words replaced by "<unk>"
        - test data with low frequent words replaced by "<unk>"
        - vocabulary of words that appear n times or more in the training data
    """
    # Get the closed vocabulary using the train data
    vocabulary = get_words_with_nplus_frequency(train_data, count_threshold)

    # For the train data, replace less common words with "<unk>"
    train_data_replaced = replace_oov_words_by_unk(train_data, vocabulary, unknown_token)

    # For the test data, replace less common words with "<unk>"
    test_data_replaced = replace_oov_words_by_unk(test_data, vocabulary, unknown_token)

    return train_data_replaced, test_data_replaced, vocabulary

# Preprocessing data
minimum_freq = 2
train_data_processed, test_data_processed, vocabulary = preprocess_data(train_data,
                                                                        test_data,
                                                                        minimum_freq)

# Save train_data and test_data
file_name = "train_data.pkl"
open_file = open(file_name, "wb")
pickle.dump(train_data_processed, open_file)
open_file.close()

file_name = "test_data.pkl"
open_file = open(file_name, "wb")
pickle.dump(test_data_processed, open_file)
open_file.close()

# Save vocabulary
file_name = "vocabulary.pkl"
open_file = open(file_name, "wb")
pickle.dump(vocabulary, open_file)
open_file.close()