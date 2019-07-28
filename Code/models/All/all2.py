import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import recall_score, confusion_matrix

from tqdm import tqdm

import json
import argparse
import sys

def train_and_test(params):


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Arguments for LSTM model')
    parser.add_argument('--train', type=str, help = "Train set file")
    parser.add_argument('--utils', type=str, help = "Helper directory")
    parser.add_argument('--db', type=str, help = "DB name", required=True)
    parser.add_argument('--collection', type=str, help = "DB collection")
    parser.add_argument('--host', type=str, help = "DB host")
    parser.add_argument('--port', type=int, help = "Port number of db")