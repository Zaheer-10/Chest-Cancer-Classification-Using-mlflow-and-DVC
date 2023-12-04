from Chest_Cancer_Classification import logger
from Chest_Cancer_Classification.components.model_evaluation_mlflow import Evaluation
from Chest_Cancer_Classification.config.configuration import ConfigurationManager
from Chest_Cancer_Classification.utils.copy_model import copy_file_to_destination


STAGE_NAME = "Model Evaluation stage"

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.log_into_mlflow()
        



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        source_file_path = "artifacts/training/model.h5"
        destination_folder_path = "model"
        copy_file_to_destination(source_file_path, destination_folder_path)
        logger.info(f">>>>>> Model Copied to Model folder from artifacts - completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e