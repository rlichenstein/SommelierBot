#Wine Recommendation Data Assembly: Robert Lichenstein, Campbell Boswell, Kyle Calder
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score
import numpy as np
import csv
import os

#just use full dataset for now
filenob = "winemag-data-130k.csv"


fields = []
rows = []
#rankings = [] #rankings dictionary
#traindata = [] #item features dictionary
def processcsv(filename):
    return csv.DictReader(open(filename, 'r', encoding="utf8"))


cvratings = processcsv('ratings_crossval.csv');
trainratings = processcsv('ratings_training.csv');
traindata = processcsv('winemag-data_training.csv');
cvdata = processcsv('winemag-data_crossval.csv');

#with open(filename, 'r',encoding="utf8") as csvfile:
#    csvreader = csv.reader(csvfile)
#    fields = next(csvreader) 
    
#    for row in csvreader:
        
#        rows.append(row)
#        rankings.append({'taster':row[12], 'rating':row[4], 'title':row[9]})
#        traindata.append({'title':row[9],'country':row[1],'province':row[6],'region_1':row[7],'variety':row[10],'winery':row[11]})

 #   print("Total # of rows = %d"%(csvreader.line_num))

#print('field names are:' + ', '.join(field for field in fields))
dataset = Dataset()
print(trainratings.fieldnames)
dataset.fit((x['taster_number'] for x in trainratings),(y['wine_id'] for y in trainratings)) #provide iterators for users and the corresponding items


#manually add all features to the dataset
dataset.fit_partial(items=(x['wine_id'] for x in traindata),item_features=(x['country'] for x in traindata))
dataset.fit_partial(items=(x['wine_id'] for x in traindata),item_features=(x['province'] for x in traindata))
dataset.fit_partial(items=(x['wine_id'] for x in traindata),item_features=(x['region_1'] for x in traindata))
dataset.fit_partial(items=(x['wine_id'] for x in traindata),item_features=(x['variety'] for x in traindata))
dataset.fit_partial(items=(x['wine_id'] for x in traindata),item_features=(x['winery'] for x in traindata))

num_users, num_items = dataset.interactions_shape()
print('Num users: {}, num_items {}.'.format(num_users, num_items))
#print the relevant information

(interactions, weights) = dataset.build_interactions(((x['taster_number'],x['wine_id']) for x in trainratings))
print(repr(interactions))

item_features = dataset.build_item_features(((x['title'],[x['country'],x['province'],x['region_1'],x['variety'],x['winery']]) for x in traindata))
print(repr(item_features))

#set up our LightFM model fit. 
model = LightFM(loss='bpr')
model.fit(interactions,item_features=item_features)
#train_precision = precision_at_k(model,traindata,k=10).mean()
predictions = model.predict(np.array([1,2,3,4,5,6,7,8]),np.array([25,26,27,28,29,30,31,32]),item_features=item_features)
print(predictions)
#TODO: Divide into train/cv data and process, then run the model. 
