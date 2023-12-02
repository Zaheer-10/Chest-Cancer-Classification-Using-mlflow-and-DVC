import os 
import sys
import logging

# logger_str_format = ['%(asctime)s : %(level)s : %(module)s : %(message)s']
logger_str_format = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"


log_dir = 'logs'
log_file_path = os.path.join(log_dir ,"running_logs.log" )

os.makedirs(log_dir , exist_ok=True)

logging.basicConfig(
    level=logging.INFO , 
    format = logger_str_format ,
    handlers=[
        logging.FileHandler(log_file_path) , 
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("Chest_Cancer_Classification_Logger")