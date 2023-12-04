import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from pathlib import Path

class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename

    def predict(self):
        
        # model = load_model(os.path.join("artifacts","training", "model.h5"))
        model = load_model(os.path.join("model", "model.h5"))
        
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224)) # converting it to an array (image.img_to_array) and resizing it to the required input size of (224, 224).
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0) # Expands the dimensions of the image array using np.expand_dims to match the expected input shape for the model.
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        # prediction = 'Normal' if result[0] == 1 else 'Adenocarcinoma Cancer'
        # print(prediction)
        # return [{ "image" : prediction}]
        if result[0] == 1:
            prediction = 'Normal'
            return [{ "image" : prediction}]
        else:
            prediction = 'Adenocarcinoma Cancer'
        
        print(prediction)    
        return [{ "image" : prediction}]
    
# obj = PredictionPipeline('artifacts/data_ingestion/Chest-CT-Scan-data/normal/2 - Copy (3).png')
# obj = PredictionPipeline('artifacts/data_ingestion/Chest-CT-Scan-data/adenocarcinoma/000015 (9).png')
# obj = PredictionPipeline('artifacts/data_ingestion/Chest-CT-Scan-data/adenocarcinoma/000019 (5).png')
# obj = PredictionPipeline('artifacts/data_ingestion/Chest-CT-Scan-data/adenocarcinoma/000040 (3).png')
# obj.predict()