import os 
import time
import tensorflow as tf
from zipfile import ZipFile
import urllib.request as request
from pathlib import Path
from Chest_Cancer_Classification.entity import TrainingConfig


class Training:
    def __init__(self , config: TrainingConfig):
        self.config = config
        
    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
        
    def train_valid_generator(self):
        '''
        `datagenerator_kwargs` is a dictionary containing the arguments for the ImageDataGenerator. It includes a rescale factor of 1/255 to scale the pixel values to the range [0, 1]. It also sets a validation split of 20%.
        '''
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )
        
        
        '''
        dataflow_kwargs is a dictionary containing the arguments for the data flow. It includes a target size for the images, a batch size, and an interpolation method of "bilinear".
        '''
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )


        ''' 
        The valid_datagenerator is created using the datagenerator_kwargs and the dataflow_kwargs. It reads the images from the training data directory and treats 20% of them as the validation subset.
        '''
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )



        ''' 
        If image augmentation is enabled (determined by the config.params_is_augmentation parameter), the train_datagenerator is created with the same parameters as the valid_datagenerator, but also includes additional augmentation parameters like rotation, flipping, and zooming. If augmentation is disabled, the train_datagenerator is set to the valid_datagenerator.
        
        The valid_generator and train_generator are created using the respective data generators. They read the images from the training data directory and create batches for the model to train on.
        
        '''
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        ''' 
        The save_model method is a static method that saves the trained model to a specified path
        '''
        model.save(path)



    
    def train(self):
        ''' 
        The train method performs the actual training of the model. It calculates the number of steps per epoch for both the training and validation data. Then, it uses the model.fit method to train the model on the training data for a specified number of epochs, with the validation data used to evaluate the model's performance during training.
        '''
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )