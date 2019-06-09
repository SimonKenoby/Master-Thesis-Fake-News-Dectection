import numpy as np
import os
import json
import pandas as pd
import logging
import argparse


import sys
sys.path.append('../..')

import utils.dbUtils
import utils.gensimUtils

from gensim.models import ldamodel
from gensim.corpora import Dictionary


if __name__ == "__main__":

	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	parser = argparse.ArgumentParser()
	parser.add_argument("topics", help="The number of topics", type=int)
	args = parser.parse_args()

	print("Creating corpus")
	
	corpus = utils.gensimUtils.corpusBuilder('news_cleaned', filters={'type' : {'$in' : ['fake', 'reliable']}})

	print("Building model")
	lda = ldamodel.LdaModel(corpus, num_topics=args.topics)
	lda.save('lda_{}'.format(args.topics))