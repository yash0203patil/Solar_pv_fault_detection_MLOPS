import os,sys
import yaml
from ultralytics import YOLO
from wasteDetection.logger import logging
from wasteDetection.exception import AppException
from wasteDetection.entity.config_entity import ModelTrainerConfig
from wasteDetection.entity.artifacts_entity import ModelTrainerArtifact,DataIngestionArtifact
from wasteDetection.entity.config_entity import DataIngestionConfig


class ModelTrainer:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.model_trainer_config = model_trainer_config


    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            logging.info("Unzipping data")
            model = YOLO(self.model_trainer_config.weight_name)
            model.train(data = os.path.join(self.data_ingestion_artifact.feature_store_path, "SOLAR_ANNOTATION" ,"data.yaml"), 
                                     epochs = self.model_trainer_config.no_epochs, 
                                     batch = self.model_trainer_config.batch_size)
           
            # os.system("cp runs/detect/train/weights/best.pt yolov8/")
            # os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            # os.system(f"cp runs/detect/train/weights/best.pt {self.model_trainer_config.model_trainer_dir}/")
           

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path="yolov8/best.pt",
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact


        except Exception as e:
            raise AppException(e, sys)


