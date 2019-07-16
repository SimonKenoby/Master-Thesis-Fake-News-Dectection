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
		self.collection.insert_one({'date' : datetime.datetime.now(), 'experiment_id' : experiment_id, 'model' : model})

	def addResult(self, json_result, experiment_id):
		self.collection.update_one({'experiment_id' : experiment_id}, {'$push' : {'result' : json_result}})

	def addEpochs(self, epoch, results, experiment_id, field_name):
		self.collection.update_one({'experiment_id' : experiment_id,  'result.epoch' : epoch}, {'$set' : {'result.$.valid' : results}})

	def addParams(self, params, experiment_id):
		self.collection.update_one({'experiment_id' : experiment_id}, {'$push' : {'params' : params}})
