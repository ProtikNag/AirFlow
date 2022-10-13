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


local_weights_file = '../inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy')>0.92):
            print("\nReached 92.0% accuracy so cancelling training!")
            self.model.stop_training = True


def get_pretrained_model():
    pre_trained = InceptionV3(
        input_shape=(150,150,3),
        include_top=False,
        weights=None
    )
    pre_trained.load_weights(local_weights_file)
    
    return pre_trained


def get_model(pre_trained, last_output):
    x = tf.keras.layers.Flatten()(last_output)
    x = tf.keras.layers.Dense(1024,activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(512,activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128,activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(6,activation='softmax')(x)        
    model = Model(inputs=pre_trained.input, outputs=x)
    
    return model


def train_model(ti):
    pre_trained = get_pretrained_model()
    last_desired_layer = pre_trained.get_layer('mixed7')
    last_output = last_desired_layer.output
    
    model = get_model(pre_trained=pre_trained, last_output=last_output)
    model.compile(
        optimizer = 'adam', 
        loss = tf.keras.metrics.categorical_crossentropy,
        metrics = ['accuracy']
    )
    
    callbacks = MyCallback()
    
    train_generator = ti.xcom_pull(key='train_generator')
    validation_generator = ti.xcom_pull(key='validation_generator')
    
    history = model.fit(
        train_generator,
        validation_data = validation_generator,
        epochs = 15,
        verbose = 2,
        callbacks=callbacks
    )
    
    ti.xcom_push(key='train_history', value=history)