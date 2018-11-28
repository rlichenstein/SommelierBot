#Wine Recommendation Data Assembly: Robert Lichenstein, Campbell Boswell, Kyle Calder
from lightfm import LightFM
from lightfm.data import Dataset
import numpy as np
import csv
import os

#just use full dataset for now
filename = "winemag-data-130k.csv"


fields = []
rows = []
rankings = [] #rankings dictionary
winefeatures = [] #item features dictionary
with open(filename, 'r',encoding="utf8") as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader) 
    
    for row in csvreader:
        
        rows.append(row)
        rankings.append({'taster':row[12], 'rating':row[4], 'title':row[9]})
        winefeatures.append({'title':row[9],'country':row[1],'province':row[6],'region_1':row[7],'variety':row[10],'winery':row[11]})

    print("Total # of rows = %d"%(csvreader.line_num))

print('field names are:' + ', '.join(field for field in fields))

dataset = Dataset()
dataset.fit((x['taster'] for x in rankings),(y['title'] for y in winefeatures)) #provide iterators for users and the corresponding items


#manually add all features to the dataset
dataset.fit_partial(item_features=(x['country'] for x in winefeatures))
dataset.fit_partial(item_features=(x['province'] for x in winefeatures))
dataset.fit_partial(item_features=(x['region_1'] for x in winefeatures))
dataset.fit_partial(item_features=(x['variety'] for x in winefeatures))
dataset.fit_partial(item_features=(x['winery'] for x in winefeatures))

num_users, num_items = dataset.interactions_shape()
print('Num users: {}, num_items {}.'.format(num_users, num_items))
#print the relevant information

(interactions, weights) = dataset.build_interactions(((x['taster'],x['title']) for x in rankings))
print(repr(interactions))

item_features = dataset.build_item_features(((x['title'],[x['country'],x['province'],x['region_1'],x['variety'],x['winery']]) for x in winefeatures))
print(repr(item_features))

#set up our LightFM model fit. 
model = LightFM(loss='bpr')
model.fit(interactions,item_features=item_features)
prediction = model.predict([1,2,3],[y['title'] for y in winefeatures])

#TODO: Divide into train/cv data and process, then run the model. 
