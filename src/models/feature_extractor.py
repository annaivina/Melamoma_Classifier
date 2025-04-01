import tensorflow as tf
from tensorflow.keras.layers import Dropout, MaxPool2D, Conv2D, BatchNormalization



class Customconv2D(tf.keras.layers.Layer):
  def __init__(self, filters, kernel_size, strides, padding='valid', l2 = 0.0, name=None):
    super(Customconv2D, self).__init__(name=name)

    self.custom_conv2d = Conv2D(filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding,
                                activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(l2)
                                )
    #Application of batch activation - befopre and after relu is debadable - needs testimg
    self.batch_norm = BatchNormalization()

  def build(self, input_shape):
    print(f"Building Customconv2D with input shape: {input_shape}")

  def call(self, input, training = True):
    x = self.custom_conv2d(input)
    x = self.batch_norm(x, training=training)
    return x
  



class FeatureExtractor(tf.keras.layers.Layer):
  def __init__(self, config):
    super(FeatureExtractor, self).__init__(name="feature_extractor")
  
    self.block_1 = tf.keras.Sequential([
                                         Customconv2D(filters=config['model_params']['cnn']['conv_1'], 
                                                      kernel_size=config['model_params']['cnn']['kernel_1'], 
                                                      strides=config['model_params']['cnn']['stride_1'], 
                                                      padding=config['model_params']['cnn']['padding1'], 
                                                      l2=config['model_params']['cnn']['l2'], name='conv1_block1'),
                                                      
                                         Customconv2D(filters=config['model_params']['cnn']['conv_1']*2, 
                                                     kernel_size=config['model_params']['cnn']['kernel_1'], 
                                                     strides=config['model_params']['cnn']['stride_1'], 
                                                     padding=config['model_params']['cnn']['padding1'], 
                                                     l2=config['model_params']['cnn']['l2'], name='conv2_block1'),
                                                     
                                         MaxPool2D(pool_size=config['model_params']['cnn']['pool_size'], strides=config['model_params']['cnn']['pool_strides'], name='maxp1_block1'),
                                         Dropout(config['model_params']['cnn']['dropout_1'],name='drop1_block1')
    ])
    
    self.block_2 = tf.keras.Sequential([
                                         Customconv2D(filters=config['model_params']['cnn']['conv_2'], 
                                                      kernel_size=config['model_params']['cnn']['kernel_2'], 
                                                      strides=config['model_params']['cnn']['stride_2'], 
                                                      padding=config['model_params']['cnn']['padding2'], 
                                                      l2=config['model_params']['cnn']['l2'], name='conv3_block2'),
                                         
                                         Customconv2D(filters=config['model_params']['cnn']['conv_2']*2, 
                                                      kernel_size=config['model_params']['cnn']['kernel_2'], 
                                                      strides=config['model_params']['cnn']['stride_2'], 
                                                      padding=config['model_params']['cnn']['padding2'], 
                                                      l2=config['model_params']['cnn']['l2'], name='conv4_block2'),
                                         MaxPool2D(pool_size=config['model_params']['cnn']['pool_size'], strides=config['model_params']['cnn']['pool_strides'], name='maxp2_block2'),
                                         Dropout(config['model_params']['cnn']['dropout_2'], name='drop2_block2')
    ])

    self.block_3 = tf.keras.Sequential([
                                         Customconv2D(filters=config['model_params']['cnn']['conv_3'], 
                                                      kernel_size=config['model_params']['cnn']['kernel_3'], 
                                                      strides=config['model_params']['cnn']['stride_3'], 
                                                      padding=config['model_params']['cnn']['padding3'], 
                                                      l2=config['model_params']['cnn']['l2'], name='conv5_block3'),

                                         Customconv2D(filters=config['model_params']['cnn']['conv_3'], 
                                                      kernel_size=config['model_params']['cnn']['kernel_3'], 
                                                      strides=config['model_params']['cnn']['stride_3'], 
                                                      padding=config['model_params']['cnn']['padding3'], 
                                                      l2=config['model_params']['cnn']['l2'], name='conv6_block3'),

                                         MaxPool2D(pool_size=config['model_params']['cnn']['pool_size'], strides=config['model_params']['cnn']['pool_strides'], name='maxp3_block3'),
                                         Dropout(config['model_params']['cnn']['dropout_3'], name='drop3_block3')
])


  def build(self, input_shape):
    print(f"Building FeatureExtractor with input shape: {input_shape}")


  def call(self, inputs, training=True):
    x = self.block_1(inputs, training=training)
    x = self.block_2(x, training=training)
    x = self.block_3(x, training=training)
    return x 