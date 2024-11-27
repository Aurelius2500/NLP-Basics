# -*- coding: utf-8 -*-
"""
Data Analytics Computing: Basics of Natural Language Processing (NLP) with Python
Spyder version 5.3.3
"""

# Import required packages 

import nltk.tokenize
import nltk.sentiment

# If nltk is not installled, then run pip and install it

# pip install nltk

# We will use this string 

string_ex = """This is a string that will help us understanding natural language processing.
This is the second sentence of the string. Computer are usually pretty bad at understanding human language.
In order to help Python to understand human language, we will need to import the nltk package, that will help us in a lot of tasks"""

# The most basic task in Python using a string is printing it, that returns it to the console
print(string_ex)

# We can also get the length of a string. By default, this is how many characters are in the string
len(string_ex)

# Notice that we can also iterate through a string as with any other iterable

type(string_ex)

for letter in string_ex:
    print(letter)

# In order to understand the meaning of text, programming languages need to understand what the text means
# One of the most basic tasks is to tokenize a string, that for this video, we can understand as splitting the meaning of the text
# sent_tokenize will divide the text into sentences, or try, while word_tokenize will do the same for words

sent = nltk.tokenize.sent_tokenize(string_ex)

print(sent)

# We get a list of sentences that we can print, get the length, or even get a specific value from the list
# We can see that the text has four sentences

len(sent)

sent[1]

sent[2]

# We can repeat the same process with words too

word = nltk.tokenize.word_tokenize(string_ex)

print(word)

len(word)

# First word of the text

word[0]

# Stop words are words that we do not usually need, like very common words with not a lot of meaning
# NLTK by default has a list of stopwords

# Run the line below if you have never installed nltk before and don't have stopwords
# nltk.download("stopwords")

stop_words = set(nltk.corpus.stopwords.words("english"))

# We can filter words using the stopwords

fil_list = []

for w in word:
    if w.casefold() not in stop_words:
        fil_list.append(w)
        
print(fil_list)

# We also have the concept of Part of Speech, that deals with the roles that words play in sentences

pos_words = nltk.pos_tag(fil_list)

print(pos_words)

# We can also get their meanings, and if you feel curious, join the meanings to the tuples

nltk.help.upenn_tagset()

freq = nltk.FreqDist(fil_list)

freq.plot(cumulative = False)

# Use all words

freq2 = nltk.FreqDist(word)

freq2.plot(cumulative = False)

# Finally, the last concept that we will see is sentiment, or how we feel about a sentence
# NLTK has a very handy sentiment analyzer, that workds pretty similar to other machine learning models that we have seen in the past

sent_analyzer = nltk.sentiment.SentimentIntensityAnalyzer()

sent_scores = sent_analyzer.polarity_scores(string_ex)

print(sent_scores)