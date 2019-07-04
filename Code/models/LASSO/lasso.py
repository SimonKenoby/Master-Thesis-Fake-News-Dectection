import numpy as np
import argparse

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load

import pandas as pd
import json

import datetime



if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Training Lasso model')
        parser.add_argument('train', type=str, help = "Train set dir")
        parser.add_argument('prefix', type=str, help = "Prefix for report dump")
        parser.add_argument('outmodel', type=str, help = "Output file for model")

        args = parser.parse_args()

        print("Loading data")
        data = pd.read_json(args.train, lines=True)
        train = data['tokenized_text'][0:64]
        y = np.array(data['type'][0:64], dtype=str)

        train = np.array([" ".join(line) for line in train])

        train_accuracy = []
        test_accuracy = []
        kf = KFold(n_splits=3, shuffle = True)
        print("Starting kfold")
        for i, (train_index, test_index) in enumerate(kf.split(y)):
                print("{}th iteration".format(i))
                model = Lasso()
                print("Fitting tfidf")
                vectorizer = TfidfVectorizer()
                X_train = vectorizer.fit_transform(train[train_index])
                X_test = vectorizer.transform(train[test_index])

                y_train = y[train_index]
                y_test = y[test_index]
                print("Training model")
                model.fit(X_train, y_train)
                train_accuracy.append(model.score(X_train, y_train))
                test_accuracy.append(model.score(X_test, y_test))
                crp = classification_report(y_test, model.predict(X_test), labels=['fake', 'reliable'], output_dict = True)

                with open(args.prefix+"{}.json".format(i), 'w') as outfile:  
                        json.dump(crp, outfile)

        model = Lasso()
        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform([' '.join(text) for text in corpus['tokenized_text']])
        model.fit(X_train, y)

        dump(model, args.outmodel) 