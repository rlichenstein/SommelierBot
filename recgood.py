#Wine Recommendation Data Assembly: Robert Lichenstein, Campbell Boswell, Kyle Calder
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, auc_score,recall_at_k
import numpy as np
import csv
import os
import sys
import itertools
import matplotlib
import matplotlib.pyplot as plt

#just use full dataset for now
filename = "winemag-data-130k.csv"


fields = []
rows = []
rankings = [] #rankings dictionary
winefeatures = [] #item features dictionary
tasterdict = {}
winedict = {}

#set up our LightFM model fit.
def randomsearch(interactions, cvinteractions, item_features, num_samples=30):
    loop = 0
    for hyperparams in itertools.islice(sample_hyperparameters(),num_samples):
        print("we're at loop " + str(loop))
        num_epochs = hyperparams.pop("num_epochs")
        model = LightFM(**hyperparams)
        model.fit(interactions,item_features=item_features,epochs=num_epochs)
        #uncomment below to maximize precision_at_k with hyperparaters
        #score = precision_at_k(model,cvinteractions,interactions,item_features=item_features).mean()
        score = auc_score(model,cvinteractions,interactions,item_features=item_features).mean()
        hyperparams["num_epochs"] = num_epochs
        loop += 1

        yield(score, hyperparams, model)


def sample_hyperparameters():
    """
    Yield possible hyperparameter choices.
    """

    while True:
        yield {
            "no_components": np.random.randint(16, 64),
            "learning_schedule": np.random.choice(["adagrad", "adadelta"]),
            "loss": np.random.choice(["bpr", "warp", "warp-kos"]),
            "learning_rate": np.random.exponential(0.05),
            "item_alpha": np.random.exponential(1e-8),
            "user_alpha": np.random.exponential(1e-8),
            "max_sampled": np.random.randint(5, 15),
            "num_epochs": np.random.randint(5, 50),
        }
def optimize_epochs(model,cvdata,traindata,item_features,epochs):
    auc = []
    for epoch in range(1,epochs):
        print('we are at epoch = ' + str(epoch))
        model.fit(interactions,item_features=item_features,epochs=epoch)
        auc.append(auc_score(model,cvdata,traindata,item_features=item_features).mean())
    x = np.arange(len(auc))
    plt.plot(x, np.array(auc))
    plt.legend(['auc score'], loc='lower right')
    plt.xlabel('num_epochs')
    plt.ylabel('auc_score',color='cadetblue')
    plt.show()
    print(max(auc))
    return auc.index(max(auc))

if __name__ == "__main__":
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
    #print(repr(interactions))

    (testinteractions, testweights) = dataset.build_interactions(((x['taster'],x['title']) for x in testrankings))
    (cvinteractions, cvweights) = dataset.build_interactions(((x['taster'],x['title']) for x in cvrankings))


    fields.remove('title')
    item_features = dataset.build_item_features((x['title'],[x[field] for field in fields[1:]]) for x in winefeatures)
    #print(repr(item_features))

    #uncomment below to run randomized optimization
    #yieldlist = list(randomsearch(interactions, cvinteractions, item_features))
    #(score, hyperparams, model) = max(yieldlist, key=lambda x: x[0])
    #print("Best score {} at {}".format(score, hyperparams))
    #print(yieldlist)

    #first best parameters
    #bestparams = {'no_components': 42, 'learning_schedule': 'adagrad', 'loss': 'warp',
    #    'learning_rate': 0.12124098084538826, 'item_alpha': 2.6444123794692915e-09,
    #    'user_alpha': 2.1437428228585198e-08, 'max_sampled': 15, 'random_state': 69}

    #second best parameters returned
    #Best score 0.9843319654464722 at
    bestparams = {'no_components': 59,
    'learning_schedule': 'adagrad', 'loss': 'warp', 'learning_rate': 0.08565020895037347,
     'item_alpha': 7.345729662383957e-10, 'user_alpha': 4.776609106732949e-09,
     'max_sampled': 14, 'random_state':69}
    model = LightFM(**bestparams)
    num_epochs = 8
    #num_epochs = optimize_epochs(model,cvinteractions,interactions,item_features,15)
    #print(num_epochs)
    model.fit(interactions,item_features=item_features,epochs=num_epochs)

    testscore = auc_score(model,testinteractions,interactions,item_features=item_features).mean()
    testprec = precision_at_k(model, testinteractions,interactions,item_features=item_features,k=10).mean()
    testrecall = recall_at_k(model, testinteractions,interactions,item_features=item_features,k=10).mean()
    print('Test AUC Score = '+str(testscore))
    print('Test Precision = '+str(testprec))
    print('Test Recall = '+str(testrecall))


    ranks = model.predict(1, np.arange(num_items))
    top_items = np.argsort(-ranks)
    #print(np.sort(ranks))
    print("Recommendations for user 1:")
    for i in range(0,11):
        print('Recommendation #' + str(i+1) + ' is: ' + winedict[str(top_items[i])])
