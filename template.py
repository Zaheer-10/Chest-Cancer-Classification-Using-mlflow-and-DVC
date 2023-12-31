import os
import logging
from pathlib import Path

# FORMAT = '%(asctime)s %(levelname)-8s %(message)s',

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = 'Chest_Cancer_Classification'

list_of_files = [

    '.github/workflows/.gitkeep' , 

    f'src/{project_name}/__init__.py',
    f'src/{project_name}/components/__init__.py',
    f'src/{project_name}/utils/__init__.py',
    f'src/{project_name}/utils/common.py',
    f'src/{project_name}/config/configuration.py',
    f'src/{project_name}/pipeline/__init__.py',
    f'src/{project_name}/entity/__init__.py',
    f'src/{project_name}/constants/__init__.py',

    'config/config.yaml',
    'dvc.yaml' , 
    'params.yaml',

    'requirements.txt' ,
    'setup.py' , 
    'RESEARCH/trials.ipynb',

    'templates/index.html',
    'static/css',
    'static/js',
    'PERSONNEL_NOTE.txt',
    

]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir , filename = os.path.split(filepath)

    if filedir !='':
        os.makedirs(filedir , exist_ok=True)
        logging.info(f'Created directory : {filedir} for the file {filename}')

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            logging.info(f'Created empty file : {filename} at location {filepath} ')
            
    else: 
        logging.info(f'{filename} is Already Exists!')
        
        