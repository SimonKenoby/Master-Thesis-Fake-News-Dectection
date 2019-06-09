import seaborn as sns
import matplotlib.pyplot as plt
from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client.TFE

word_count = []

for obj in db.reliable.find({'word_count' : {'$lt' : 1000}}):
    word_count.append(obj['word_count'])
    
sns.boxplot(y = word_count)