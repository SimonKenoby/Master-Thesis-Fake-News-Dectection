import pandas as pd
import gensim
from gensim.corpora import Dictionary
import argparse

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Training Lasso model')
	parser.add_argument('inFile', type=str, help = "File containing the texts")
	parser.add_argument('outFile', type=str, help = "File for saving the dictionary")

	args = parser.parse_args()
	data = pd.read_json(args.inFile, lines = True)
	dct = Dictionary(data['tokenized_text'])

	dct.save(args.outFile)