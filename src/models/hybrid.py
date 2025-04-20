import tensorflow as tf


class HybridModel(tf.keras.models.Model):
    def __init__(self, vit_model, cnn_model, weight_path, input_shape, freeze_cnn=True):
        
        cnn_model.compile(optimizer='adam', 
                          loss=tf.keras.losses.BinaryFocalCrossentropy(label_smoothing=0.0, gamma=3),
                          metrics=['accuracy']) #Compile CNN just for loading werights... 
        
        cnn_model(tf.zeros((1,) + input_shape), training=False)
        cnn_model.load_weights(weight_path)

        self.feature_extractor = cnn_model.FeatureExtractor
        self.feature_extractor.trainable = not freeze_cnn

        self.vit_model = vit_model

        self.model = self.build_model(input_shape)

    def build_model(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        x = self.feature_extractor(inputs)
        outputs = self.vit_model(x)

        return tf.keras.Model(inputs,outputs)
    
    def call(self, inputs, training=False):
        return self.model(inputs, training=training)