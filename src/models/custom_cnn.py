import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,Flatten, MaxPool2D
import os
from .feature_extractor import FeatureExtractor


class MelanomaClassifier(tf.keras.Model):

    def __init__(self, config):
        super(MelanomaClassifier, self).__init__(name='customNN')

        self.feature_extractor = FeatureExtractor(config)
        self.flatten = Flatten()
        self.dense_1 = Dense(config['model_params']['dense']['dense_1'], activation='relu',name='dense1')
        self.dense_2 = Dense(config['model_params']['dense']['dense_2'], activation='relu', name='dense2')
        self.output_layer =  Dense(config['dataset']['num_classes'], activation=config['model_params']['dense']['activation'], name='out')


    def call(self, input, training = True):
        x = self.feature_extractor(input, training=training)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dense_2(x)

        return self.output_layer(x)

  




    
