import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.sentiment.components.data_transformation import DataTransformation
from src.sentiment.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "sentiment/train.csv")
    test_data_path: str = os.path.join('artifacts', "sentiment/test.csv")
    raw_data_path: str = os.path.join('artifacts', "sentiment/data.tsv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebooks\sentiment\data\data.tsv', delimiter='\t', quoting=3)
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    # step 2: data transformation
    data_transformation_obj = DataTransformation()

    X_train_scaled, X_test_scaled, y_train, y_test = data_transformation_obj.initiate_data_transformation(train_data,
                                                                                                          test_data)

    logging.info("Data transformation completed successfully")

    # Step 3: Model Training
    model_trainer_obj = ModelTrainer()
    accuracy, cross_val_mean, cross_val_std = model_trainer_obj.initiate_model_trainer(X_train_scaled, X_test_scaled,
                                                                                       y_train, y_test)

    logging.info(f"Final Accuracy: {accuracy}")
    logging.info(f"Cross-validation Mean Accuracy: {cross_val_mean}")
    logging.info(f"Cross-validation Standard Deviation: {cross_val_std}")
