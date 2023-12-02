import os
import gdown
import zipfile
import logging
from Chest_Cancer_Classification import logger
from Chest_Cancer_Classification.entity import DataIngestionConfig
from Chest_Cancer_Classification.utils.common import get_size

class DataIngestion:
    def __init__(self , config: DataIngestionConfig):
        self.config = config
        
    def download_file(self) -> str:
        '''
        Fetch the dataset from the google drive (url)
        '''
        try:
            dataset_url = self.config.source_url
            zip_dir = self.config.local_data_file
            os.makedirs('artifacts/data_ingestion' , exist_ok=True)
            logger.info(f'msg=Downloading data from {dataset_url} into file {zip_dir} (zip file)')
            
            file_id = dataset_url.split('/')[-2]
            prefix = 'https://drive.google.com/u/0/uc?/export=download&id='
            
            
            gdown.download(prefix+file_id, zip_dir)
            
            logger.info(f'Downloaded data from {dataset_url} into file {zip_dir} (zip file)')
            
        except Exception as e:
            raise e
        
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
            