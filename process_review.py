"""
SommelierBot
Campbell Boswell, Rober Lichenstein, Kyle Calder
CS 451 Final Project
process_review.py
"""
import csv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import math
from nltk.probability import FreqDist
import keyword_filter

def import_reviews():
    '''
    A simple function which imports review text from the kaggel data set we are
    working with. This function also removes stopwords from the reviews using
    the stopword list included in NLTK
    '''

    print('importing reviews from csv [', end='',flush=True)
    review_list = []        #a list of all the reviews we process
    stop_words = set(stopwords.words('english'))

    with open('winemag-data-130k.csv', mode='r', encoding='utf8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row = next(csv_reader) #skip the first row which contains lables
        i = 0
        for row in csv_reader:

            if (i % 1000) == 0:
                print('*', end='',flush=True)
            review_as_list = []     #a single review in list form
            review_as_text = row[2]

            word_tokens = word_tokenize(review_as_text)
            for w in word_tokens:
                if w not in stop_words:
                    if len(w) > 3:
                        review_as_list.append(w.lower())

            review_list.append(' '.join(review_as_list))
            i += 1

        print(']')

    return review_list


def build_word_dict(review_list, frequency):
    '''
    Builds a dictionary of the n most common terms across our reviews
    (where n is specified by the variable "frequency"). This list of terms
    is composed by removing some of the most common words that are
    non-adjectives and reflect the sampling of wine in gerneral instead of the
    flavor profile of a specific wine. These words that are flagged for removal
    are the file keyword_filter.py
    '''

    print('building word dictionary [', end='', flush=True)

    keyword_dict = {} #a dictionary of keywords and indices to be returned
    filter_list = keyword_filter.filter_list #a manually formed list of words to filter out

    #concatenate all the reviews to calculate the frequency of terms in all reviews
    review_blob = ''.join(review_list)
    words = word_tokenize(review_blob) #tokenize words in review blob
    freq_dist = FreqDist(words) #get the frequency distributions of all words
    keywords = freq_dist.most_common(frequency) # create a list of the most common words

    print("keywords length: " + str(len(keywords)))
    keyword_count = 0
    i = 0

    while i < len(keywords):

        if keywords[i][0] in filter_list:
            del keywords[i]
        else:
            i += 1
        keyword_count += 1

    #Last == ('great', 4820)
    print(keywords)
    print("Initial dict length: " + str(frequency) + " new dict length: " + str(i))


    i = 0
    for word in keywords:
        if i > 272:
            print(word)
        #print progess
        if (i % 10) == 0:
            print('*', end='', flush=True)

        keyword_dict.update({ word[0] : i})
        i += 1

    print(']')
    return keyword_dict


def assign_features(keyword_dict, review_list):
    '''
    This function creates a content vector for each review that reflects the
    presence of any of the most common words listed in our dictionary generated
    by the function build_word_dict. These content vectors can be written to an
    output csv with the function write_features
    '''
    print('generating frequency scores [', end='', flush=True)
    feature_count = len(keyword_dict)
    feature_list = []
    bloblist = []

    i = 0
    #itterate through reviews and mark instanes of feature words
    for review in review_list:
        #print progess
        if (i % 1000) == 0:
            print('*', end='', flush=True)

        #create our feature vector
        features = [0] * feature_count

        word_tokens = word_tokenize(review)
        for w in word_tokens:
            if w in keyword_dict:
                index = keyword_dict.get(w)

                #insert the score at the proper coresponding index
                features[index] +=1

        feature_list.append(features)
        i+=1

    print(']')
    return feature_list


def write_features(feature_list, keyword_dict):
    '''
    This function writes the features we generated across all data to a csv so
    that it can be easily fed to our learning algorithm.
    '''
    print("writing fetures to csv")
    with open('review_features.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')

        #write label row for keywords
        keyword_list = []
        for keyword in keyword_dict:
            keyword_list.append(keyword)
        writer.writerow(keyword_list)

        for features in feature_list:
            writer.writerow(features)



def main():
    review_list = import_reviews()
    keyword_dict = build_word_dict(review_list, 285)
    feature_list = assign_features(keyword_dict, review_list)
    write_features(feature_list, keyword_dict)

    #Debugging output to ensure that we hitting features in all documents
    print("We have " + str(len(feature_list[0])) + " features...")

    feature_set = set()
    num_empty_docs = 0
    for features in feature_list:
        doc_feature_count = 0
        for i in range(len(features)):
            if features[i] != 0:
                feature_set.add(i)
                doc_feature_count += 1
        if doc_feature_count == 0:
            num_empty_docs += 1

    print("feature_count: " + str(len(feature_set)))
    print("num_empty_docs: " +str(num_empty_docs))



if __name__ == "__main__":
    main()
