import tensorflow as tf
from tensorflow.keras.layers import Dropout, MaxPool2D, Conv2D, BatchNormalization



class Customconv2D(tf.keras.layers.Layer):
  def __init__(self, filters, kernel_size, strides, padding='valid', l2 = 0.0, name=None):
    super(Customconv2D, self).__init__(name=name)

    self.filters = filters
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.l2 = l2
    self.name = name

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

  def get_config(self):
    config = super().get_config()
    config.update({"filters": self.filters,
                   "kernel_size": self.kernel_size,
                   "strides": self.strides,
                   "padding": self.padding,
                   "l2": self.l2,
                   "name": self.name})
    return config

  @classmethod
  def from_config(cls, config):
    return cls(**config)
  

  



class FeatureExtractor(tf.keras.layers.Layer):
  def __init__(self, cfg):
    super(FeatureExtractor, self).__init__(name="feature_extractor")

    self.cfg = cfg 
  
    self.block_1 = tf.keras.Sequential([
                                         Customconv2D(filters=cfg.model.cnn.conv_1, 
                                                      kernel_size=cfg.model.cnn.kernel_1, 
                                                      strides=cfg.model.cnn.stride_1, 
                                                      padding=cfg.model.cnn.padding1, 
                                                      l2=cfg.model.cnn.l2, name='conv1_block1'),
                                                      
                                         Customconv2D(filters=cfg.model.cnn.conv_1*2, 
                                                     kernel_size=cfg.model.cnn.kernel_1, 
                                                     strides=cfg.model.cnn.stride_1, 
                                                     padding=cfg.model.cnn.padding1, 
                                                     l2=cfg.model.cnn.l2, name='conv2_block1'),
                                                     
                                         MaxPool2D(pool_size=cfg.model.cnn.pool_strides, strides=cfg.model.cnn.pool_strides, name='maxp1_block1'),
                                         Dropout(cfg.model.cnn.dropout_1,name='drop1_block1')
    ])
    
    self.block_2 = tf.keras.Sequential([
                                         Customconv2D(filters=cfg.model.cnn.conv_2, 
                                                      kernel_size=cfg.model.cnn.kernel_2, 
                                                      strides=cfg.model.cnn.stride_2, 
                                                      padding=cfg.model.cnn.padding2, 
                                                      l2=cfg.model.cnn.l2, name='conv3_block2'),
                                         
                                         Customconv2D(filters=cfg.model.cnn.conv_2*2, 
                                                      kernel_size=cfg.model.cnn.kernel_2, 
                                                      strides=cfg.model.cnn.stride_2, 
                                                      padding=cfg.model.cnn.padding2, 
                                                      l2=cfg.model.cnn.l2, name='conv4_block2'),
                                         MaxPool2D(pool_size=cfg.model.cnn.pool_size, strides=cfg.model.cnn.pool_strides, name='maxp2_block2'),
                                         
                                         Dropout(cfg.model.cnn.dropout_2, name='drop2_block2')
    ])

    self.block_3 = tf.keras.Sequential([
                                         Customconv2D(filters=cfg.model.cnn.conv_3, 
                                                      kernel_size=cfg.model.cnn.kernel_3, 
                                                      strides=cfg.model.cnn.stride_3, 
                                                      padding=cfg.model.cnn.padding3, 
                                                      l2=cfg.model.cnn.l2, name='conv5_block3'),

                                         Customconv2D(filters=cfg.model.cnn.conv_3, 
                                                      kernel_size=cfg.model.cnn.kernel_3, 
                                                      strides=cfg.model.cnn.stride_3, 
                                                      padding=cfg.model.cnn.padding3, 
                                                      l2=cfg.model.cnn.l2, name='conv6_block3'),

                                         MaxPool2D(pool_size=cfg.model.cnn.pool_size, strides=cfg.model.cnn.pool_strides, name='maxp3_block3'),
                                         
                                         Dropout(cfg.model.cnn.dropout_3, name='drop3_block3')
])


  def build(self, input_shape):
    print(f"Building FeatureExtractor with input shape: {input_shape}")


  def call(self, inputs, training=True):
    x = self.block_1(inputs, training=training)
    x = self.block_2(x, training=training)
    x = self.block_3(x, training=training)
    return x 
  

  def get_config(self):
    config = super().get_config()
    config.update({"config": self.cfg,
                   "block_1": self.block_1,
                   "block_2": self.block_2,
                   "block_3": self.block_3,})
    return config

  @classmethod
  def from_config(cls, config):
    return cls(**config)
  
