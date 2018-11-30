#Wine Recommendation Data Assembly: Robert Lichenstein, Campbell Boswell, Kyle Calder
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, auc_score
import numpy as np
import csv
import os

#just use full dataset for now
filename = "winemag-data-130k.csv"


fields = []
rows = []
rankings = [] #rankings dictionary
winefeatures = [] #item features dictionary
tasterdict = {}
winedict = {}
with open(filename, 'r',encoding="utf8") as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    fields.remove('description')
    fields.remove('designation')
    fields.remove('taster_name')
    fields.remove('taster_number')
    wineid = 0 
    for row in csvreader:
        
        rows.append(row)
        rankings.append({'taster':row[12], 'points':row[4], 'title':row[9]})
        winefeatures.append({'title':row[9],'country':row[1],'province':row[6],'region_1':row[7],'variety':row[10],'winery':row[11],'points':row[4], 'price': row[5]})
        for i in range(9, 209):
            winefeatures[wineid][str(fields[i])] = row[i]
        if (row[12] not in tasterdict):
                if(row[12] == 20):
                    tasterdict[row[12]] = 'John Doe'
                else:
                    tasterdict[row[12]] = row[8]
        if (str(wineid) not in winedict):
            winedict[str(wineid)] = row[9]
        wineid+= 1
            
    #print("Total # of rows = %d"%(csvreader.line_num))

#print('field names are:' + ', '.join(field for field in fields))


trainrankings = rankings[0:83183]
cvrankings = rankings[83184:101007]
testrankings = rankings[101008:]

trainfeats = winefeatures[0:90980]
cvfeats = winefeatures[90980:110476]
testfeats = winefeatures[110476:]
dataset = Dataset()
dataset.fit((x['taster'] for x in trainrankings),(y['title'] for y in winefeatures)) #provide iterators for users and the corresponding items


#manually add all features to the dataset
dataset.fit_partial(item_features=(x['country'] for x in winefeatures))
dataset.fit_partial(item_features=(x['province'] for x in winefeatures))
dataset.fit_partial(item_features=(x['region_1'] for x in winefeatures))
dataset.fit_partial(item_features=(x['variety'] for x in winefeatures))
dataset.fit_partial(item_features=(x['winery'] for x in winefeatures))
dataset.fit_partial(item_features=(x['points'] for x in winefeatures))
dataset.fit_partial(item_features=(x['price'] for x in winefeatures))
for i in range(9,209):
    dataset.fit_partial(item_features=(x[str(fields[i])] for x in winefeatures))

num_users, num_items = dataset.interactions_shape()
#print('Num users: {}, num_items {}.'.format(num_users, num_items))
#print the relevant information

(interactions, weights) = dataset.build_interactions(((x['taster'],x['title']) for x in trainrankings))
print(repr(interactions))

(testinteractions, testweights) = dataset.build_interactions(((x['taster'],x['title']) for x in testrankings))

#item_features = dataset.build_item_features(((x['title'],[x['country'],x['province'],x['region_1'],x['variety'],x['winery']]) for x in winefeatures))
fields.remove('title')
item_features = dataset.build_item_features((x['title'],[x[field] for field in fields[1:]]) for x in winefeatures)
print(repr(item_features))

#set up our LightFM model fit. 
model = LightFM(loss='warp', random_state= 69,max_sampled = 100)
model.fit(interactions,item_features=item_features,epochs=50)
predictionmatrix = model.predict_rank(testinteractions,interactions,item_features=item_features)
#print(repr(predictionmatrix))
print('precision = '  + str(precision_at_k(model,testinteractions,interactions,item_features=item_features).mean()))
print('ROC AUC METRIC = ' + str(auc_score(model,testinteractions,interactions,item_features=item_features).mean()))

ranks = model.predict(1, np.arange(num_items))
top_items = np.argsort(-ranks)
#print(top_items)
#print(np.sort(ranks))
for i in range(0,11):
    print('Recommendation #' + str(i+1) + ' is: ' + winedict[str(top_items[i])])

