from pymongo import MongoClient
import argparse
import sys

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Arguments for connecting to the DB')
	parser.add_argument('--db', type=str, help = "DB name", required=True)
	parser.add_argument('--collection', type=str, help = "DB collection")
	parser.add_argument('--host', type=str, help = "DB host")
	parser.add_argument('--port', type=int, help = "Port number of db")

	args = parser.parse_args()

	client = MongoClient(args.host, args.port)
	db = client[args.db]
	collection = db[args.collection]

	out = collection.find({'experiment_id' : {"$exists" : True}}, {'experiment_id' : 1}).sort('experiment_id', -1).limit(1)

	sys.exit(out[0]['experiment_id'] + 1)
