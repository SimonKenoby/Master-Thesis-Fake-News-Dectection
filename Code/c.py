from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client.TFE
collection = db.news
collection2 = db.news_copy

res = collection2.find()

for obj in res:
	collection.update_one({'_id' : obj['_id']}, {'$set' : {'avgSentenceLength' : obj['avgSentenceLength'], 'numSentences' : obj['numSentences']}})
