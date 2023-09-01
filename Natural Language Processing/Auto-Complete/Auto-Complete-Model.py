import math
import random
import numpy as np
import pandas as pd
import nltk
import re
import pickle
nltk.download('punkt')
nltk.data.path.append('.')
import streamlit as st

# Open train_data
train_file = "train_data.pkl"
open_file = open(train_file, "rb")
train_data_processed = pickle.load(open_file)
open_file.close()

# Open test_data
test_file = "test_data.pkl"
open_file = open(test_file, "rb")
test_data_processed = pickle.load(open_file)
open_file.close()

# Open vocabulary file
vocabulary_file = "vocabulary.pkl"
open_file = open(vocabulary_file, "rb")
vocabulary = pickle.load(open_file)
open_file.close()

def count_n_grams(data, n, start_token='<s>', end_token='<e>'):
    """
    Count all n-grams in the data

    Args:
        data: List of lists of words
        n: number of words in a sequence

    Returns:
        A dictionary that maps a tuple of n-words to its frequency
    """

    # Initialize dictionary of n-grams and their counts
    n_grams = {}

    # Go through each sentence in the data
    for sentence in data:  # complete this line

        # prepend start token n times, and  append the end token one time
        sentence = [start_token for i in range(n)] + sentence + [end_token]

        # convert list to tuple
        # So that the sequence of words can be used as
        # a key in the dictionary
        sentence = tuple(sentence)

        # Use 'i' to indicate the start of the n-gram
        # from index 0
        # to the last index where the end of the n-gram
        # is within the sentence.

        for i in range(len(sentence) - n + 1):  # complete this line

            # Get the n-gram from i to i+n
            n_gram = sentence[i:i + n]

            # check if the n-gram is in the dictionary
            if n_gram in n_grams:  # complete this line with the proper condition

                # Increment the count for this n-gram
                n_grams[n_gram] += 1
            else:
                # Initialize this n-gram count to 1
                n_grams[n_gram] = 1

    return n_grams


def estimate_probability(word, previous_n_gram,
                         n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
    """
    Estimate the probabilities of a next word using the n-gram counts with k-smoothing

    Args:
        word: next word
        previous_n_gram: A sequence of words of length n
        n_gram_counts: Dictionary of counts of n-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary_size: number of words in the vocabulary
        k: positive constant, smoothing parameter

    Returns:
        A probability
    """
    # convert list to tuple to use it as a dictionary key
    previous_n_gram = tuple(previous_n_gram)

    # Set the denominator
    # If the previous n-gram exists in the dictionary of n-gram counts,
    # Get its count.  Otherwise set the count to zero
    # Use the dictionary that has counts for n-grams
    if previous_n_gram not in n_gram_counts:
        previous_n_gram_count = 0
    else:
        previous_n_gram_count = n_gram_counts[previous_n_gram]

    # Calculate the denominator using the count of the previous n gram
    # and apply k-smoothing
    denominator = previous_n_gram_count + k * vocabulary_size

    # Define n plus 1 gram as the previous n-gram plus the current word as a tuple
    n_plus1_gram = previous_n_gram + (word,)

    # Set the count to the count in the dictionary,
    # otherwise 0 if not in the dictionary
    # use the dictionary that has counts for the n-gram plus current word
    if n_plus1_gram in n_plus1_gram_counts:
        n_plus1_gram_count = n_plus1_gram_counts[n_plus1_gram]
    else:
        n_plus1_gram_count = 0

    # Define the numerator use the count of the n-gram plus current word,
    # and apply smoothing
    numerator = n_plus1_gram_count + k

    # Calculate the probability as the numerator divided by denominator
    probability = numerator / denominator

    return probability


def estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, end_token='<e>',
                           unknown_token="<unk>", k=1.0):
    """
    Estimate the probabilities of next words using the n-gram counts with k-smoothing

    Args:
        previous_n_gram: A sequence of words of length n
        n_gram_counts: Dictionary of counts of n-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary: List of words
        k: positive constant, smoothing parameter

    Returns:
        A dictionary mapping from next words to the probability.
    """
    # convert list to tuple to use it as a dictionary key
    previous_n_gram = tuple(previous_n_gram)

    # add <e> <unk> to the vocabulary
    # <s> is not needed since it should not appear as the next word
    vocabulary = vocabulary + [end_token, unknown_token]
    vocabulary_size = len(vocabulary)

    probabilities = {}
    for word in vocabulary:
        probability = estimate_probability(word, previous_n_gram,
                                           n_gram_counts, n_plus1_gram_counts,
                                           vocabulary_size, k=k)

        probabilities[word] = probability

    return probabilities


def calculate_perplexity(sentence, n_gram_counts, n_plus1_gram_counts, vocabulary_size, start_token='<s>',
                         end_token='<e>', k=1.0):
    """
    Calculate perplexity for a list of sentences

    Args:
        sentence: List of strings
        n_gram_counts: Dictionary of counts of n-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary_size: number of unique words in the vocabulary
        k: Positive smoothing constant

    Returns:
        Perplexity score
    """
    # length of previous words
    n = len(list(n_gram_counts.keys())[0])

    # prepend <s> and append <e>
    sentence = [start_token] * n + sentence + [end_token]

    # Cast the sentence from a list to a tuple
    sentence = tuple(sentence)

    # length of sentence (after adding <s> and <e> tokens)
    N = len(sentence)

    # The variable p will hold the product
    # that is calculated inside the n-root
    # Update this in the code below
    product_pi = 1.0

    # Index t ranges from n to N - 1, inclusive on both ends
    for t in range(n, N):
        # get the n-gram preceding the word at position t
        n_gram = sentence[t - n:t]

        # get the word at position t
        word = sentence[t]

        # Estimate the probability of the word given the n-gram
        # using the n-gram counts, n-plus1-gram counts,
        # vocabulary size, and smoothing constant
        probability = estimate_probability(word, n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k)

        # Update the product of the probabilities
        # This 'product_pi' is a cumulative product
        # of the (1/P) factors that are calculated in the loop
        product_pi *= 1 / probability
        ### END CODE HERE ###

    # Take the Nth root of the product
    perplexity = (product_pi) ** (1 / N)

    return perplexity


def suggest_a_word(previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, end_token='<e>',
                   unknown_token="<unk>", k=1.0, start_with=None):
    """
    Get suggestion for the next word

    Args:
        previous_tokens: The sentence you input where each token is a word. Must have length >= n
        n_gram_counts: Dictionary of counts of n-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary: List of words
        k: positive constant, smoothing parameter
        start_with: If not None, specifies the first few letters of the next word

    Returns:
        A tuple of
          - string of the most likely next word
          - corresponding probability
    """

    # length of previous words
    n = len(list(n_gram_counts.keys())[0])

    # append "start token" on "previous_tokens"
    previous_tokens = ['<s>'] * n + previous_tokens

    # From the words that the user already typed
    # get the most recent 'n' words as the previous n-gram
    previous_n_gram = previous_tokens[-n:]

    # Estimate the probabilities that each word in the vocabulary
    # is the next word,
    # given the previous n-gram, the dictionary of n-gram counts,
    # the dictionary of n plus 1 gram counts, and the smoothing constant
    probabilities = estimate_probabilities(previous_n_gram,
                                           n_gram_counts, n_plus1_gram_counts,
                                           vocabulary, k=k)

    # Initialize suggested word to None
    # This will be set to the word with highest probability
    suggestion = None

    # Initialize the highest word probability to 0
    # this will be set to the highest probability
    # of all words to be suggested
    max_prob = 0

    # For each word and its probability in the probabilities dictionary:
    for word, prob in probabilities.items():  # complete this line

        # If the optional start_with string is set
        if start_with is not None:  # complete this line with the proper condition

            # Check if the beginning of word does not match with the letters in 'start_with'
            if word.startswith(start_with) == False:  # complete this line with the proper condition

                # if they don't match, skip this word (move onto the next word)
                continue

        # Check if this word's probability
        # is greater than the current maximum probability
        if prob > max_prob:  # complete this line with the proper condition

            # If so, save this word as the best suggestion (so far)
            suggestion = word

            # Save the new maximum probability
            max_prob = prob

    return suggestion, max_prob


def get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0, start_with=None):
    model_counts = len(n_gram_counts_list)
    suggestions = []
    for i in range(model_counts - 1):
        n_gram_counts = n_gram_counts_list[i]
        n_plus1_gram_counts = n_gram_counts_list[i + 1]

        suggestion = suggest_a_word(previous_tokens, n_gram_counts,
                                    n_plus1_gram_counts, vocabulary,
                                    k=k, start_with=start_with)
        suggestions.append(suggestion)
    return suggestions

# Calculate n_gram_counts_list
n_gram_counts_list = []
for n in range(1, 6):
    print("Computing n-gram counts with n =", n, "...")
    n_model_counts = count_n_grams(train_data_processed, n)
    n_gram_counts_list.append(n_model_counts)

st.title("Auto-Complete")
previous_sentence = st.text_input("Input your sentence ...")
if previous_sentence:
    previous_tokens = nltk.word_tokenize(previous_sentence)
    tmp_suggest = get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0)
    words = [word for word, prob in tmp_suggest]
    words = set(words)
    st.write("The suggestion words are: ", words)

