# SommelierBot
A Wine Recommender system for use by wine enthusiasts and reviewers. Implemented using lightfm and a Wine Review dataset from Kaggle

The following details the functionality of files created during the development
of SommelierBot:

process_review.py
    A python script which produces content vectors from the text-based wine reviews in the original kaggle data set. This program removes stopwords from the review corpus then calculates the 200 most popular strings across all reviews, ignoring any string that is listed in the primary list stored in keyword_filter.py. Each review is then scored to produce a content vector by individually totaling the instances of words in the set of 200 most common words. The content vectors that are generated during this process are output to a csv.

keyword_filter.py
    This file contains a list of just over 100 words which are to be ignored during the creation of a set of the 200 most common words across all reviews. The words are ignored because they either are extremely general, but were not removed from the reviews when filtering for stopwords, or because they describe the wine tasting process but not aspects of the wine itself (i.e. glass, flavors, mouth).
    
recgood.py
    Our main python file for running the LightFM Model and generating recommendations on a per user basis. Uses asynchronous stochastic gradient descent and matrix factorization via LightFM to give us recommendations on wines for users in our dataset. Edit the file to change the output to predict for other users besides user 1. Recommended to read through the comments carefully before trying to use the either the random search or optimization functions. 

wine-data-130k.csv
 - A copy of the original Kaggle data set that has been updated to include the content vectors produced from our review text pre-processing.
