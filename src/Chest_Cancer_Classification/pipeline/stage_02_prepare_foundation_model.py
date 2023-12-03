from Chest_Cancer_Classification import logger
from Chest_Cancer_Classification.components.prepare_foundation_model import PrepareFoundationModel
from Chest_Cancer_Classification.config.configuration import ConfigurationManager


STAGE_NAME = "Prepare Foundation model"

class PrepareFoundationModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_foundation_model()
        prepare_base_model = PrepareFoundationModel(config=prepare_base_model_config)
        prepare_base_model.get_foundation_model()
        prepare_base_model.update_base_model()




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareFoundationModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e