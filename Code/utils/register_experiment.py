from pymongo import MongoClient
import argparse
import sys
import datetime


class Register():
	def __init__(self, host, port, db, collection):
		self.client = MongoClient(host, port)
		self.db = self.client[db]
		self.collection = self.db[collection]

	def newExperiment(self, experiment_id, model):
		res = self.collection.find_one({'experiment_id' : experiment_id})
		if res != None:
			self.collection.delete_one({'experiment_id' : experiment_id})
		self.collection.insert_one({'date' : datetime.datetime.now(), 'experiment_id' : experiment_id, 'model' : model})

	def addResult(self, json_result, experiment_id):
		self.collection.update_one({'experiment_id' : experiment_id}, {'$push' : {'result' : json_result}})

	def addEpochs(self, epoch, results, experiment_id, field_name):
		self.collection.update_one({'experiment_id' : experiment_id,  'result.epoch' : epoch}, {'$set' : {'result.$.valid' : results}})

	def addParams(self, params, experiment_id):
		self.collection.update_one({'experiment_id' : experiment_id}, {'$push' : {'params' : params}})

	def getLastExperiment(self):
		try:
			out = self.collection.find({'experiment_id' : {"$exists" : True}, 'finish' : True}, {'experiment_id' : 1}).sort('experiment_id', -1).limit(1)
			return out[0]['experiment_id']
		except:
			return 0

	def closeExperiment(self, experiment_id):
		self.collection.update_one({'experiment_id' : experiment_id}, {"$set" : {'finish' : True}})
