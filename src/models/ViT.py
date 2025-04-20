import tensorflow as tf 
from .transformer import TransformEncoder, MLP

class PatchEmbed(tf.keras.layers.Layer):
  def __init__(self, embed_dim, **kwargs):
    super().__init__(**kwargs)

    self.embed_dim = embed_dim
    self.proj = tf.keras.layers.Dense(embed_dim)
    self.pos_embed = None

  def build(self, input_shape):
    self.n_patches = input_shape[1]*input_shape[2]
    #Make the dinamic embedding given we dont know the n_patches
    self.pos_embed = self.add_weight(shape=(1, self.n_patches, self.embed_dim),
                                     initializer=tf.keras.initializers.RandomNormal(stddev=0.2),
                                     trainable=True,
                                     name='pos_embed')


  def call(self, input):
    batch_size = tf.shape(input)[0]
    patches = tf.reshape(input, (batch_size, tf.shape(input)[1] * tf.shape(input)[2], tf.shape(input)[3]))#reshapes to batch_size , nPatches, embed_dim to behave as tokens
    x = self.proj(patches)
    return x + self.pos_embed
  


class VIT(tf.keras.models.Model):
  def __init__(self, cfg, **kwargs):
    super().__init__(**kwargs)

    self.cfg = cfg

    self.patch_embed = PatchEmbed(embed_dim=cfg.transf.embed_dim)
    self.transf = [TransformEncoder(n_heads=cfg.transf.n_heads, embed_dim=cfg.transf.embed_dim, dropout=cfg.transf.dropout_trans) for _ in range(cfg.transf.n_layers)]

    self.mlp_out = MLP(hidden_dim=cfg.transf.n_dense_units, output_dim=cfg.transf.n_dense_units, dropout_rate=cfg.transf.dropout_mlp_out)
    self.avrg_pool = tf.keras.layers.GlobalAveragePooling1D()
    self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')


  def call(self, input):
    x = self.patch_embed(input)

    for i in range(self.cfg.transf.n_layers):
      x = self.transf[i](x)

    x = self.avrg_pool(x)
    x = self.mlp_out(x)
    output = self.output_layer(x)

    return output