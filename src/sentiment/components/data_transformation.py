import sys
import re
import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

STOPWORDS = set(stopwords.words('english'))


@dataclass
class DataTransformationConfig:
    count_vectorizer_path = os.path.join('Models', 'countVectorizer.pkl')
    scaler_path = os.path.join('Models', 'scaler.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def record_cleaning(self, data):
        try:
            logging.info("Dropping null records")
            data.dropna(inplace=True)

            logging.info("Adding length column to the dataset")
            data['length'] = data['verified_reviews'].apply(len)

            return data
        except Exception as e:
            raise CustomException(e, sys)

    def generate_wordcloud(self, reviews):

        logging.info("Generate a WordCloud object.")
        try:
            wc = WordCloud(background_color='white', max_words=50)
            wc.generate(reviews)
            return wc
        except Exception as e:
            raise CustomException(e, sys)

    def find_unique_words(self, data):
        logging.info("Finding unique words from reviews")
        try:
            neg_reviews = " ".join([review for review in data[data['feedback'] == 0]['verified_reviews']])
            neg_reviews = neg_reviews.lower().split()

            pos_reviews = " ".join([review for review in data[data['feedback'] == 1]['verified_reviews']])
            pos_reviews = pos_reviews.lower().split()

            unique_negative = " ".join([x for x in neg_reviews if x not in pos_reviews])
            unique_positive = " ".join([x for x in pos_reviews if x not in neg_reviews])

            return unique_negative, unique_positive

        except Exception as e:
            raise CustomException(e, sys)

    def preprocess_reviews(self, data):
        logging.info("Preprocessing reviews")
        try:
            corpus = []
            stemmer = PorterStemmer()
            for i in range(0, data.shape[0]):
                review = re.sub('[^a-zA-Z]', ' ', data.iloc[i]['verified_reviews'])
                review = review.lower().split()
                review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
                corpus.append(' '.join(review))
            return corpus

        except Exception as e:
            raise CustomException(e, sys)

    def vectorize_reviews(self, corpus):
        logging.info("Vectorizing reviews using CountVectorizer")
        try:
            cv = CountVectorizer(max_features=2500)
            X = cv.fit_transform(corpus).toarray()

            logging.info(f"Saving Count Vectorizer to {self.data_transformation_config.count_vectorizer_path}")
            save_object(self.data_transformation_config.count_vectorizer_path, cv)

            return X
        except Exception as e:
            raise CustomException(e, sys)

    def split_data(self, X, y):
        logging.info("Splitting data into train and test sets")
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)
            logging.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
            logging.info(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise CustomException(e, sys)

    def scale_data(self, X_train, X_test):
        logging.info("Scaling data using MinMaxScaler")
        try:
            scaler = MinMaxScaler()
            X_train_scl = scaler.fit_transform(X_train)
            X_test_scl = scaler.transform(X_test)

            logging.info(f"Saving Scaler to {self.data_transformation_config.scaler_path}")
            save_object(self.data_transformation_config.scaler_path, scaler)

            return X_train_scl, X_test_scl
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            # Loading data
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            # Applying transformations
            train_df = self.record_cleaning(train_df)

            corpus = self.preprocess_reviews(train_df)
            X = self.vectorize_reviews(corpus)
            y = train_df['feedback'].values  # feedback is the target column

            # Splitting and scaling the data
            X_train, X_test, y_train, y_test = self.split_data(X, y)
            X_train_scaled, X_test_scaled = self.scale_data(X_train, X_test)

            return X_train_scaled, X_test_scaled, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)
