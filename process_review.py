"""
def import_reviews(): - Import Wine Reviews from csv remove stopwords and
   store each review as a textblob "document" in a list of docs (or bloblist)
def build_word_dict(review_list): -
        1. Itterate through documents
            a. itterate through each word in the document and build a dictionary
               of words that contain index values (just populate in order encountered)
def review_tf_idf(keyword_dict, review_list):
    1. Itterate through documents and convert to textblobs, append the blobs to
       a list of blobs
    1. Itterate throught the list of blobs
        a. initialize a list for each document/blob that has a size that is
           equal to the number key value pairs in the dictionary
        b. itterate through each word in document/blob and calculate tf-idf score
        c. store the tf-idf score in the list we just made accourding to the
           index listed in the dictionary
        d. append the feature list to a master list of all examples
"""
import csv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import math
from nltk.probability import FreqDist


def import_reviews():
    print('importing reviews from csv [', end='',flush=True)
    review_list = []        #a list of all the reviews we process
    stop_words = set(stopwords.words('english'))

    with open('winemag-data-130k.csv', mode='r', encoding='utf8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row = next(csv_reader) #skip the first row which contains lables
        i = 0
        for row in csv_reader:

            #for efficient debugging
            #if i > 1000:
            #    break

            if (i % 1000) == 0:
                print('*', end='',flush=True)
            review_as_list = []     #a single review in list form
            review_as_text = row[2]

            #might need to double check that this field isn't empty

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
    print('building word dictionary [', end='', flush=True)
    keyword_dict = {} # a dictionary of keywords and indices to be returned

    #concatenate all the reviews to calculate the frequency of terms in all reviews
    review_blob = ''.join(review_list)
    words = word_tokenize(review_blob) #tokenize words in review blob
    freq_dist = FreqDist(words) #get the frequency distributions of all words
    #print(freq_dist)
    keywords = freq_dist.most_common(frequency) # create a list of the most common words

    i = 0
    for word in keywords:
        #print progess
        if (i % 10) == 0:
            print('*', end='', flush=True)

        keyword_dict.update({ word[0] : i})
        i += 1

    print(']')
    return keyword_dict


def assign_features(keyword_dict, review_list):
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


def write_features(feature_list):
    '''Write the features we generated across all data to a csv so that is can
       be easily fed to our learning algorithm'''
    print("writing fetures to csv")
    with open('review_features.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for features in feature_list:
            writer.writerow(features)



def main():
    review_list = import_reviews()
    keyword_dict = build_word_dict(review_list, 1000)
    feature_list = assign_features(keyword_dict, review_list)
    write_features(feature_list)



if __name__ == "__main__":
    main()
