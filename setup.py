import os
import setuptools 

with open ('README.md' , 'r' , encoding='utf-8') as f:
    long_description = f.read()
    
__version__ = '0.0.0'
REPO_NAME ='Chest-Cancer-Classification-Using-mlflow-and-DVC'
AUTHOR_USERNAME='Zaheer-10'
SRC_REPO ='Chest_Cancer_Classification'
AUTHOR_EMAIL ='zaheer.work24@gmail.com'

setuptools.setup(
    
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USERNAME,
    description="A project to classify chest cancer using mlflow and DVC",
    long_description= 'text/markdown',
    url=f"https://github.com/{AUTHOR_USERNAME}/{REPO_NAME}",
    project_urls={
        'Bug_Tracker' : f"https://github.com/{AUTHOR_USERNAME}/{REPO_NAME}/issues",
        
    },
    package_dir={"": "src"} ,
    packages = setuptools.find_packages(where='src')
    
)

