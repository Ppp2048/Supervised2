import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exceptions import CustomException

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        try:
            logging.info("Starting training pipeline")

            # Step 1: Data Ingestion
            logging.info("Running data ingestion")
            raw_data_path = self.data_ingestion.download_stock_data()

            # Step 2: Data Transformation
            logging.info("Running data transformation")
            processed_data_path = self.data_transformation.transform_data()

            # Step 3: Model Training
            logging.info("Running model training")
            model_path, metrics = self.model_trainer.train_models()

            logging.info("Training pipeline completed successfully")
            logging.info(f"Model saved at: {model_path}")
            logging.info(f"Best model: {metrics['best_model']}")

            return model_path, metrics

        except Exception as e:
            logging.error("Error in training pipeline")
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
