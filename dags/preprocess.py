from airflow.settings import AIRFLOW_HOME

import tensorflow as tf 
from tensorflow.keras import (
    layers,
    Model,
)
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img,
)
import os
from tensorflow.keras.applications.inception_v3 import InceptionV3


train_data_path = './data/train/'
validation_data_path = './data/validation/'
test_data_path = './data/test/'


def preprocess(ti):
    print("LKSDF: ",  os.getcwd())
    os.chdir("/home/protiknag")
    print("New: ", os.getcwd())
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=.2,
        zoom_range=.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )


    train_generator = train_datagen.flow_from_directory(
        directory="./Desktop/MLOps_Pipeline/AirFlowDocker/data/train",
        batch_size=32, 
        class_mode='categorical',
        target_size=(150, 150)
    )
        
    validation_datagen = ImageDataGenerator(
        rescale=1.0/255.,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        directory="./Desktop/MLOps_Pipeline/AirFlowDocker/data/validation",
        batch_size=32, 
        class_mode='categorical',
        target_size=(150, 150)
    )
    
    ti.xcom_push(key='train_generator', value=train_generator)
    ti.xcom_push(key='validation_generator', value=validation_generator)