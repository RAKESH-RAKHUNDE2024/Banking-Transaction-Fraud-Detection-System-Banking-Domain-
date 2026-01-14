from transaction.components.data_ingestion import DataIngestion
from transaction.components.data_transformation import DataTransformation
from transaction.components.model_trainer import ModelTrainer
from transaction.logger import logger
from transaction.utils import read_csv
from transaction.config import TRAIN_FILE_PATH, TEST_FILE_PATH

class TrainingPipeline:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def run(self):
        logger.info("========== TRAINING PIPELINE STARTED ==========")

        ingestion = DataIngestion(self.data_path)
        train_path, test_path = ingestion.initiate_data_ingestion()

        train_df = read_csv(train_path)
        test_df = read_csv(test_path)

        transformation = DataTransformation()
        preprocessor, X_train, y_train, X_test, y_test = transformation.transform_data(train_df, test_df)

        trainer = ModelTrainer()
        trainer.train(preprocessor, X_train, y_train, X_test, y_test)

        logger.info("========== TRAINING PIPELINE COMPLETED ==========")