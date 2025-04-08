import tensorflow as tf 
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D



class EfficientNetBase(tf.keras.Model):
    def __init__(self,cfg):
        super(EfficientNetBase,self).__init__(name='EffNetCustom')
        self.feature_extractor = tf.keras.applications.EfficientNetB2(include_top=False, weights='imagenet')
        self.output_layer = tf.keras.layers.Dense(1, activation=cfg.model.dense.activation,name='out')
        self.arv_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = tf.keras.layers.Dropout(cfg.model.dense.dropout)


    def unfreeze_top_layers(self, num_layers=10):
        #Unfreeze the layers in the EffNetB2 
        #Make sure you do not unfreeze the BatchNorm layers 
        for layer in self.feature_extractor.layers[-num_layers:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False


    def call(self, input, training=False):
        x = self.feature_extractor(input, training=training)
        x = self.arv_pool(x)
        x = self.dropout(x)
        return self.output_layer(x)
    
