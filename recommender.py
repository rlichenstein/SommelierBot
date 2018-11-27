from lightfm import LightFM
from lightfm.data import Dataset
import numpy as np
import csv
import os

filename = "winemag-data-130k.csv"


fields = []
rows = []
rankings = []
winefeatures = []
with open(filename, 'r',encoding="utf8") as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader) 
    
    for row in csvreader:
        
        rows.append(row)
        rankings.append({'taster':row[14], 'rating':row[4], 'title':row[11]})
        winefeatures.append({'title':row[11],'country':row[1],'province':row[6],'region_1':row[7],'variety':row[12],'winery':row[13]})

    print("Total # of rows = %d"%(csvreader.line_num))

print('field names are:' + ', '.join(field for field in fields))

dataset = Dataset()
dataset.fit((x['taster'] for x in rankings),(y['title'] for y in winefeatures))
num_users, num_items = dataset.interactions_shape()
print('Num users: {}, num_items {}.'.format(num_users, num_items))

