import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,Flatten, MaxPool2D
import os
from .feature_extractor import FeatureExtractor


class MelanomaClassifier(tf.keras.Model):

    def __init__(self, cfg):
        super(MelanomaClassifier, self).__init__(name='customNN')

        self.feature_extractor = FeatureExtractor(cfg)
        #self.flatten = Flatten() seems like globaverage pool is more reccomnded since it doesnt remove spacial info
        self.glob_avr_pool = GlobalAveragePooling2D()
        self.dense_1 = Dense(cfg.model.dense.dense_1, activation='relu',name='dense1')
        self.dense_2 = Dense(cfg.model.dense.dense_2, activation='relu', name='dense2')
        self.output_layer =  Dense(cfg.num_classes, activation=cfg.model.dense.activation, name='out')


    def call(self, input, training = True):
        x = self.feature_extractor(input, training=training)
        x = self.glob_avr_pool(x)
        x = self.dense_1(x)
        x = self.dense_2(x)

        return self.output_layer(x)

  




    
