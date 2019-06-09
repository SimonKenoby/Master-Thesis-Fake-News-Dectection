import pandas as pd
import numpy as np
import pymongo
import csv
from pymongo import MongoClient

#num_text = 2807968
num_text = 9408908

csv.field_size_limit(100000000)
csv.field_size_limit()

client = MongoClient('localhost', 27017)
db = client.TFE
collection = db.news

df = pd.read_csv('../../Data/news_cleaned_2018_02_13.csv', nrows=2, index_col=0)
cols_name = df.columns.values
cols_name[0] = '_id'


csv_iter = pd.read_csv('../../Data/news_cleaned_2018_02_13.csv', iterator=True, skiprows=1, chunksize=10000, index_col=0, names=cols_name, parse_dates=['scraped_at', 'inserted_at', 'updated_at'], error_bad_lines = False, engine='python')
for df in csv_iter:
    try:
        df = df.dropna(how='any', subset=['type', 'content'])
        df = df.drop_duplicates()
        df = df.loc[df.index.dropna()]
        df.drop(df[df.scraped_at.isnull()].index, inplace=True)
        doc = df.to_dict('records')
        if len(doc) == 0:
            continue
        collection.insert_many(doc)
    except KeyError:
        doc = df.to_dict('records')
        if len(doc) == 0:
            continue
        collection.insert_many(doc)
