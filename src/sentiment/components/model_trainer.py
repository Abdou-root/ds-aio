import os
import sys
from dataclasses import dataclass
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier
)

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models, evaluate_models2


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "sentiment\sentiment_model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            logging.info("Starting model training")

            models = {
                "Random Forest": RandomForestClassifier(),
                "Logistic Regression": LogisticRegression(),
                "XGBoost": XGBClassifier(),
                "CatBoost": CatBoostClassifier(verbose=False),
                "AdaBoost": AdaBoostClassifier(),
            }

            params = {
                "Random Forest": {
                    'n_estimators': [100, 300],
                    'max_depth': [80, 100],
                    'min_samples_split': [8, 12],
                },
                "XGBoost": {
                    'n_estimators': [100, 300],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5, 7]
                },
                "CatBoost": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1, 0.5]
                },
                "Logistic Regression": {
                    'C': [0.01, 0.1, 1, 10],
                    'solver': ['lbfgs', 'liblinear']
                }
            }

            model_report: dict = evaluate_models2(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            # Extract accuracy values to find the best model
            best_model_name = max(model_report, key=lambda k: model_report[k]['accuracy'])
            best_model_score = model_report[best_model_name]['accuracy']
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No good model found with acceptable accuracy")

            logging.info(f"Best model found: {best_model_name}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Predict on the test data
            y_preds = best_model.predict(X_test)

            # Compute accuracy
            accuracy = accuracy_score(y_test, y_preds)
            logging.info(f"Accuracy on test data: {accuracy}")

            # Confusion matrix
            cm = confusion_matrix(y_test, y_preds)
            cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
            cm_display.plot()
            plt.show()

            # Perform cross-validation
            cross_val_acc = cross_val_score(best_model, X_train, y_train, cv=10)
            logging.info(f"Cross-validation mean accuracy: {cross_val_acc.mean()}")
            logging.info(f"Cross-validation standard deviation: {cross_val_acc.std()}")

            return accuracy, cross_val_acc.mean(), cross_val_acc.std()

        except Exception as e:
            raise CustomException(e, sys)