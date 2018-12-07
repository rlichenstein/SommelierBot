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

#below is our dataset after the text processing and feature additions
filename = "winemag-data-130k.csv"

#main instance variables
fields = [] #csv headers
rows = [] #rows in csv
rankings = [] #rankings dictionary
winefeatures = [] #item features dictionary
tasterdict = {} #dictionary of taster names
winedict = {} #dictionary of wineids to wine names

#below performs a random search over our sampled hyper parameters, and yields the score (AUC or precision)
#for the performance of the model trained on one sample of hyperparameters
#(taken from the response of the creator of LightFm to a StackOverflow question on 
#optimizing LightFm Models, linked in paper.
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



#Simply yields hyperparameter choices for a lightfm model, randomly selected. Taken from the
#same source as above. Linked in paper.
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

#A small function to return the best choice of epochs for training (that yields the highest AUC score)
#over a range of epochs, for a lightFM model
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
    #plots the output for better visualization of an elbow
    print(max(auc))
    #returns the maximum score's epoch
    return auc.index(max(auc))

if __name__ == "__main__":
    #Below is the main data extraction which involves quite a lot of processing of the csv.
    #for many reasons we couldn't use a simpler library to extract it all into a list, but mainly
    #because we needed some specific formats for the interaction matrices LightFM uses to build a Dataset
    #object.
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
            #rankings and feature matrices selection
            rankings.append({'taster':row[12], 'points':row[4], 'title':row[9]})
            winefeatures.append({'title':row[9],'country':row[1],'province':row[6],'region_1':row[7],'variety':row[10],'winery':row[11],'points':row[4], 'price': row[5]})
            #the below just adds to the wine feature matrix all of the additional 200 word features
            #garnered from the post processing, so it needed to be done iteratively.
            for i in range(9, 209):
                winefeatures[wineid][str(fields[i])] = row[i]
            if (row[12] not in tasterdict):
                    if(row[12] == 20):
                        #assigning our blank entries to a reviewer John Doe
                        tasterdict[row[12]] = 'John Doe'
                    else:
                        #generating the taster name dictionary
                        tasterdict[row[12]] = row[8]
            if (str(wineid) not in winedict):
                #generating the wine name dictionary
                winedict[str(wineid)] = row[9]
            #iterating our wineid counter, because lightfm ID's each wine by the order
            #they are processed in the model, not necessarily thes supplied title
            wineid+= 1


    #dividing the rankings into Train/Cv/Test
    trainrankings = rankings[0:83183]
    cvrankings = rankings[83184:101007]
    testrankings = rankings[101008:]

    #dividing the features into Train/Cv/Test
    #unused currently but usable later so it is being kept in
    trainfeats = winefeatures[0:90980]
    cvfeats = winefeatures[90980:110476]
    testfeats = winefeatures[110476:]

    #LightFm Dataset Object
    dataset = Dataset()
    dataset.fit((x['taster'] for x in trainrankings),(y['title'] for y in winefeatures))
    #it needs to be fit by providing iterators for users and the corresponding items


    #manually add all features to the dataset
    dataset.fit_partial(item_features=(x['country'] for x in winefeatures))
    dataset.fit_partial(item_features=(x['province'] for x in winefeatures))
    dataset.fit_partial(item_features=(x['region_1'] for x in winefeatures))
    dataset.fit_partial(item_features=(x['variety'] for x in winefeatures))
    dataset.fit_partial(item_features=(x['winery'] for x in winefeatures))
    dataset.fit_partial(item_features=(x['points'] for x in winefeatures))
    dataset.fit_partial(item_features=(x['price'] for x in winefeatures))
    #then add our word vector features iteratively
    for i in range(9,209):
        dataset.fit_partial(item_features=(x[str(fields[i])] for x in winefeatures))

    num_users, num_items = dataset.interactions_shape()
    
    #building the interaction matrix for training ratings
    (interactions, weights) = dataset.build_interactions(((x['taster'],x['title']) for x in trainrankings))

    #and the corresponding sparse matrices for CV and Test ratings
    (testinteractions, testweights) = dataset.build_interactions(((x['taster'],x['title']) for x in testrankings))
    (cvinteractions, cvweights) = dataset.build_interactions(((x['taster'],x['title']) for x in cvrankings))

    #here we need to remove title so our next iterator works properly
    fields.remove('title')
    #double list comprehension to build the item features in a smart way, providing each feature in wine features
    #which is >200
    item_features = dataset.build_item_features((x['title'],[x[field] for field in fields[1:]]) for x in winefeatures)

    #uncomment below to run randomized optimization
    #yieldlist = list(randomsearch(interactions, cvinteractions, item_features))
    #(score, hyperparams, model) = max(yieldlist, key=lambda x: x[0])
    #print("Best score {} at {}".format(score, hyperparams))
    #print(yieldlist)
    

    #Below are the results of our random optimiaztion, hardcoded as parameters now.
    #Best score 0.9843319654464722 at
    bestparams = {'no_components': 59,
    'learning_schedule': 'adagrad', 'loss': 'warp', 'learning_rate': 0.08565020895037347,
     'item_alpha': 7.345729662383957e-10, 'user_alpha': 4.776609106732949e-09,
     'max_sampled': 14, 'random_state':69}
    model = LightFM(**bestparams)
    num_epochs = 8

    #uncomment below to optimize the number of epochs we train
    #num_epochs = optimize_epochs(model,cvinteractions,interactions,item_features,15)

    model.fit(interactions,item_features=item_features,epochs=num_epochs)

    testscore = auc_score(model,testinteractions,interactions,item_features=item_features).mean()
    testprec = precision_at_k(model, testinteractions,interactions,item_features=item_features,k=10).mean()
    testrecall = recall_at_k(model, testinteractions,interactions,item_features=item_features,k=10).mean()
    print('Test AUC Score = '+str(testscore))
    print('Test Precision = '+str(testprec))
    print('Test Recall = '+str(testrecall))

    #below we have our model predict for a User between 1 and 20. Simply change
    #the first argument to get predictions for other users.
    ranks = model.predict(1, np.arange(num_items))
    top_items = np.argsort(-ranks)
    print("Recommendations for user 1:")
    for i in range(0,11):
        print('Recommendation #' + str(i+1) + ' is: ' + winedict[str(top_items[i])])
