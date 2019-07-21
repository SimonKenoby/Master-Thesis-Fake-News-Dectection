import argparse
from pymongo import MongoClient
import sys

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Arguments for Attention model')
	parser.add_argument('--SEQ_LENGTH', type = int, help = "Fixed size length to expand or srink text")
	parser.add_argument('--EMBEDDING_DIM', type = int, help = "Size of the embedding dimention")
	parser.add_argument('--HIDDEN', type = int, help = "Size of the hidden layer")
	parser.add_argument('--LAYERS', type = int, help = "Number of hidden layers")
	parser.add_argument('--DROPOUT', type = float, help = "Number of hidden layers")
	parser.add_argument('--EPOCHS', type = int, help = "Batch size")
	parser.add_argument('--Name', type=str, help = "Model name")
	parser.add_argument('--db', type=str, help = "DB name", required=True)
	parser.add_argument('--collection', type=str, help = "DB collection")
	parser.add_argument('--host', type=str, help = "DB host")
	parser.add_argument('--port', type=int, help = "Port number of db")

	args = parser.parse_args()

	client = MongoClient(args.host, args.port)
	db = client[args.db]
	collection = db[args.collection]
	r = collection.find_one({'model' : args.Name, 'params.SEQ_LENGTH' : args.SEQ_LENGTH, 'params.EMBEDDING_DIM' : args.EMBEDDING_DIM, 'params.HIDDEN' : args.HIDDEN, 'params.LAYERS' : args.LAYERS, 'params.DROPOUT' : args.DROPOUT, 'params.EPOCHS' : args.EPOCHS, 'finish' : True})
	if r != None:
		sys.exit(1)
	else:
		sys.exit(0)