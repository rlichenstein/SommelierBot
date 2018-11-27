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
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import math


def import_reviews():
    print('importing reviews from csv')
    review_list = []        #a list of all the reviews we process
    stop_words = set(stopwords.words('english'))

    with open('winemag-data-130k-v2.csv', mode='r', encoding='utf8') as csv_file:
        print('opened the csv')
        csv_reader = csv.reader(csv_file, delimiter=',')
        row = next(csv_reader) #skip the first row which contains lables
        i = 0
        for row in csv_reader:

            if i > 1000:
                break

            print('reading document #' + str(i) +' out of 100')
            review_as_list = []     #a single review in list form
            review_as_text = row[2]

            #might need to double check that this field isn't empty

            word_tokens = word_tokenize(review_as_text)
            for w in word_tokens:
                if w not in stop_words:
                    review_as_list.append(w.lower())

            review_list.append(' '.join(review_as_list))
            i += 1

        print('finished reading csv')

    return review_list


def build_word_dict(review_list):
    print('building word dictionary')
    keyword_dict = {} #dictionary to store all keywords across all docs
    i = 0 #value to calculate coresponding indexes for feature list we are going to generate - these will be store in the keyword_dict

    for review in review_list:
        word_tokens = word_tokenize(review)
        for w in word_tokens:
            if w not in keyword_dict:
                keyword_dict.update({ w : i})
                i += 1

    print(str(keyword_dict))
    return keyword_dict


def review_tf_idf(keyword_dict, review_list):
    print('generating tf-idf scores')
    feature_count = len(keyword_dict)
    feature_list = []
    bloblist = []

    #convert reviews in to textblobs and store them in a list of textblobs
    for review in review_list:
        bloblist.append(TextBlob(review))

    #itterate through blobs and compute tf-idf scores for each word
    for blob in bloblist:
        features = [0] * feature_count
        for word in blob.words:
            score = tf_idf(word, blob, bloblist)
            index = keyword_dict.get(word)
            if index == None:

                print("ERROR, " + word + " NOT IN DICTIONARY")
                continue
            features[index] = score
        feature_list.append(features)

    return feature_list

def tf_idf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))


def main():
    review_list = import_reviews()
    keyword_dict = build_word_dict(review_list)
    feature_list = review_tf_idf(keyword_dict, review_list)
    print(str(feature_list[0]))

    feature_set = set()
    num_empty_docs = 0
    for features in feature_list:
        doc_feature_count = 0
        for i in range(len(features)):
            if features[i] >= 0.1:
                feature_set.add(i)
                doc_feature_count += 1
        if doc_feature_count == 0:
            num_empty_docs += 1

    print("feature_count: " + str(len(feature_set)))
    print("num_empty_docs: " +str(num_empty_docs))



if __name__ == "__main__":
    main()
