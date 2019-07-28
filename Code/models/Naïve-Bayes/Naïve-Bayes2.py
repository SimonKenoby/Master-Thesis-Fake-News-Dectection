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
import pandas as pd


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Arguments for LSTM model')
    parser.add_argument('--train', type=str, help = "Train set file")
    parser.add_argument('--test', type=str, help = "Test set file")
    parser.add_argument('--utils', type=str, help = "Helper directory")
    parser.add_argument('--max_features', type=int, help = "Max Feature for tf-idf")
    parser.add_argument('--dataset', type=str, help = "Name for the dataset.")
    parser.add_argument('--db', type=str, help = "DB name", required=True)
    parser.add_argument('--collection', type=str, help = "DB collection")
    parser.add_argument('--host', type=str, help = "DB host")
    parser.add_argument('--port', type=int, help = "Port number of db")

    args = parser.parse_args()

    print("Loading data")

    train = pd.read_json(args.train)
    test = pd.read_json(args.test)

    sys.path.append(args.utils)

    from register_experiment import Register

    r = Register(args.host, args.port, args.db, args.collection)
    r.newExperiment(r.getLastExperiment() + 1, 'Naïve-Bayes')

    train = utils.dbUtils.TokenizedIterator('liar-liar', filters = {'split' : 'train'})
    y_train = np.array([x for x in train.iterTags()])

    test = utils.dbUtils.TokenizedIterator('news_cleaned', filters = {'split' : 'valid'})
    y_test = np.array([x for x in test.iterTags()])

    print("Fiting tf-idf")

    vectorizer = TfidfVectorizer(max_features = args.max_features)		
    X_train = vectorizer.fit_transform([' '.join(news) for news in train])	
    X_test = vectorizer.transform([' '.join(news) for news in test])

    model = MultinomialNB()
    model.fit(X_train, y_train)

    crp = classification_report(y_test, model.predict(X_test), labels=['fake', 'reliable'], output_dict = True)


    r.addResult({
    	'train' :
    			{'clr' : classification_report(y_train, model.predict(X_train), labels=['fake', 'reliable'], output_dict = True), 'confusion matrix' : confusion_matrix(y_train, model.predict(X_train))}}
    	'test' : 
    			{'clr' : clr, 'confusion matrix' : confusion_matrix(y_test, model.predict(X_test))},
    			r.getLastExperiment() + 1)
    r.addParams({'max_features' : args.max_features, 'dataset' : args.dataset}, r.getLastExperiment() + 1)
    r.closeExperiment(r.getLastExperiment() + 1)
