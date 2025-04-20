import tensorflow as tf 



class MLP(tf.keras.layers.Layer):
  def __init__(self, hidden_dim, output_dim, dropout_rate, **kwargs):
    super().__init__(**kwargs)

    self.dense_1 = tf.keras.layers.Dense(hidden_dim, activation='gelu')
    self.dense_2 = tf.keras.layers.Dense(output_dim)
    self.dropout = tf.keras.layers.Dropout(dropout_rate)


  def call(self, input):
    x = self.dense_1(input)
    x = self.dense_2(x)
    x = self.dropout(x)

    return x
  


class TransformEncoder(tf.keras.layers.Layer):
  def __init__(self, n_heads, embed_dim, dropout, **kwargs):
    super().__init__(**kwargs)

    self.n_heads = n_heads
    self.embed_dim = embed_dim

    #2 norm layers
    self.norm_1 = tf.keras.layers.LayerNormalization()
    self.norm_2 = tf.keras.layers.LayerNormalization()
    self.multi_head = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=embed_dim//n_heads)
    self.mlp = MLP(hidden_dim=embed_dim, output_dim=embed_dim, dropout_rate=dropout)


  def call(self, input):
    x_1 = self.norm_1(input)
    x_1 = self.multi_head(x_1, x_1) #pass two because key is assumed as value
    x_1 = tf.keras.layers.Add()([x_1, input])
    x_2 = self.norm_2(x_1)
    output = self.mlp(x_2)
    output = tf.keras.layers.Add()([output, x_1])

    return output
  