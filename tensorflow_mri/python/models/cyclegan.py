# Copyright 2021 University College London. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.s
# ==============================================================================

# pylint: disable=arguments-differ

"""CycleGAN Layers and Composite Generator/Discriminator
Model Implementation."""

import tensorflow as tf

from keras.activations import sigmoid
from keras.initializers import RandomNormal
from keras.layers import Layer
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Conv2DTranspose
from keras.layers import Activation
from keras.layers import Concatenate
from keras.models import Model
from keras.optimizers import Adam

from tensorflow_addons.layers import InstanceNormalization

class CGResNetBlock(Layer):
  """ResNet Block Layer specific to the CycleGAN Generator Architecture."""
  def __init__(self, n_filters):
    """
    Args:
      n_filters: An `int` denoting the number of convolutional filters
      to use, and same dtype as `self`.
    """
    super().__init__()
    self.init = RandomNormal(stddev=0.02)
    self.conv2a = Conv2D(n_filters, (3,3), padding="same",
        kernel_initializer=self.init)
    self.conv2b = Conv2D(256, (3,3), padding="same",
        kernel_initializer=self.init)
    self.instance_norm1 = InstanceNormalization(axis=-1)
    self.instance_norm2 = InstanceNormalization(axis=-1)
    self.relu = Activation("relu")
    self.concat = Concatenate()

  def call(self, inputs):
    x = self.conv2a(inputs)
    x = self.instance_norm1(x)
    x = self.relu(x)
    x = self.conv2b(x)
    x = self.instance_norm2(x)
    x = self.concat([x, inputs])
    return x

class CGEncoder(Layer):
  """Encoder Layer specific to the CycleGAN Generator Architecture."""

  def __init__(self):
    super().__init__()
    self.init = RandomNormal(stddev=0.02)
    self.conv2a = Conv2D(64, (7,7), padding='same',
        kernel_initializer=self.init)
    self.conv2b = Conv2D(128, (3,3), strides=(2,2), padding='same',
        kernel_initializer=self.init)
    self.conv2c = Conv2D(256, (3,3), strides=(2,2), padding='same',
        kernel_initializer=self.init)
    self.instance_norm1 = InstanceNormalization(axis=-1)
    self.instance_norm2 = InstanceNormalization(axis=-1)
    self.instance_norm3 = InstanceNormalization(axis=-1)
    self.relu = Activation('relu')

  def call(self, inputs):
    x = self.conv2a(inputs)
    x = self.instance_norm1(x)
    x = self.relu(x)

    x = self.conv2b(x)
    x = self.instance_norm2(x)
    x = self.relu(x)

    x = self.conv2c(x)
    x = self.instance_norm3(x)
    x = self.relu(x)

    return x

class CGDecoder(Layer):
  """Decoder Layer specific to the CycleGAN Generator Architecture."""
  def __init__(self):
    super().__init__()
    self.init = RandomNormal(stddev=0.02)
    self.conv2ta = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same',
        kernel_initializer=self.init)
    self.conv2tb = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same',
        kernel_initializer=self.init)
    self.conv2c = Conv2D(1, (7,7), padding='same', kernel_initializer=self.init)
    self.instance_norm1 = InstanceNormalization(axis=-1)
    self.instance_norm2 = InstanceNormalization(axis=-1)
    self.instance_norm3 = InstanceNormalization(axis=-1)
    self.tanh = Activation('tanh')
    self.relu = Activation('relu')

  def call(self, inputs):
    x = self.conv2ta(inputs)
    x = self.instance_norm1(x)
    x = self.relu(x)

    x = self.conv2tb(x)
    x = self.instance_norm2(x)
    x = self.relu(x)

    x = self.conv2c(x)
    x = self.instance_norm3(x)
    out_image = self.tanh(x)

    return out_image

class CGConvEncoder(Layer):
  """Convolutional Encoder Layer specific to the
  CycleGAN Discriminator Architecture."""
  def __init__(self):
    super().__init__()
    self.init = RandomNormal(stddev=0.02)

    self.conv2a = Conv2D(64, (4,4), strides=(3,3), padding='same',
        kernel_initializer=self.init)
    self.conv2b = Conv2D(128, (4,4), strides=(3,3), padding='same',
        kernel_initializer=self.init)
    self.conv2c = Conv2D(256, (4,4), strides=(3,3), padding='same',
        kernel_initializer=self.init)
    self.conv2d = Conv2D(512, (4,4), strides=(2,2), padding='same',
        kernel_initializer=self.init)
    self.conv2e = Conv2D(512, (4,4), padding='same',
        kernel_initializer=self.init)
    self.conv2f = Conv2D(1, (4,4), padding='same',
        kernel_initializer=self.init)

    self.instance_norm1 = InstanceNormalization(axis=-1)
    self.instance_norm2 = InstanceNormalization(axis=-1)
    self.instance_norm3 = InstanceNormalization(axis=-1)
    self.instance_norm4 = InstanceNormalization(axis=-1)

    self.leakyrelu = LeakyReLU(alpha=0.2)
    self.sigmoid = sigmoid

  def call(self, inputs):
    """
    Args:
      inputs: A `tf.Tensor` of shape `[..., n]` and same dtype as `self`.
    Returns:
      A `tf.Tensor` of shape `[..., 1]` and same dtype as `self`.
    """
    x = self.conv2a(inputs)
    x = self.leakyrelu(x)

    x = self.conv2b(x)
    x = self.instance_norm1(x)
    x = self.leakyrelu(x)

    x = self.conv2c(x)
    x = self.instance_norm2(x)
    x = self.leakyrelu(x)

    x = self.conv2d(x)
    x = self.instance_norm3(x)
    x = self.leakyReLU(x)

    x = self.conv2e(x)
    x = self.instance_norm4(x)
    x = self.leakyrelu(x)

    out_pred = self.conv2f(x)

    return out_pred

class CGGenerator(Model):
  """CycleGAN Generator Model."""
  def __init__(self, image_shape, n_resnet=9):
    super().__init__()

    # If image size is less than 256x256, only use
    # 6 resnet blocks
    if (image_shape[1] < 256 and n_resnet is None) or n_resnet is None:
      n_resnet = 6

    self.n_resnet = n_resnet
    self.encoder = CGEncoder()
    self.decoder = CGDecoder()
    self.res_blocks = []

    for _ in range(0, self.n_resnet):
      self.res_blocks.append(CGResNetBlock(256))

  def call(self, inputs):
    x = self.encoder(inputs)
    for block in self.res_blocks:
      x = block(x)
    x = self.decoder(x)
    return x

class CGDiscriminator(Model):
  """CycleGAN Discriminator Model."""
  def __init__(self):
    super().__init__()
    self.conv_encoder = CGConvEncoder()

  def compile(self, optimizer=Adam(lr=0.0002, beta_1=0.5),
  loss_weights=(0.5), **kwargs):
    super().compile(optimizer=optimizer, losds_weights=loss_weights **kwargs)

  def call(self, inputs):
    return self.conv_encoder(inputs)

class CycleGAN(Model):
  """
  This is the main composite CycleGAN  Model used to concurrently
  train both generators and discriminators with a custom training loop.

  Adapted from the following: https://keras.io/examples/generative/cyclegan/

  By default, the image size for both Generators and Discriminators
  is set to be 128x128 - but these can be changed to any square image
  dimension, although > 256x256 and < 64x64 is not recommended.
  """

  def __init__(
    self,
    g_loss_fn,
    d_loss_fn,
    adversarial_loss_fn,
    identity_loss_fn,
    generator_g = CGGenerator(image_shape=(1,128,128)),
    generator_f = CGGenerator(image_shape=(1,128,128)),
    discriminator_x = CGDiscriminator(),
    discriminator_y = CGDiscriminator(),
    lambda_cycle=10.0,
    lambda_identity=1.0,
  ):
    super().__init__()
    self.gen_g = generator_g
    self.gen_f = generator_f
    self.disc_x = discriminator_x
    self.disc_y = discriminator_y
    self.lambda_cycle = lambda_cycle
    self.lambda_identity = lambda_identity

    self.g_loss_fn = g_loss_fn
    self.d_loss_fn = d_loss_fn
    self.adv_loss_fn = adversarial_loss_fn
    self.identity_loss_fn = identity_loss_fn

  # TODO - Move these to a tutorial notebook and add them
  # as input to the cycleGAN as they are now not needed
  # within the class here.

  # Define the loss function for the generators
  #def generator_loss_fn(self, fake):
  #  fake_loss = self.adv_loss_fn(tf.ones_like(fake), fake)
  #  return fake_loss

  # Define the loss function for the discriminators
  #def discriminator_loss_fn(self, real, fake):
  #  real_loss = self.adv_loss_fn(tf.ones_like(real), real)
  #  fake_loss = self.adv_loss_fn(tf.zeros_like(fake), fake)
  #  return (real_loss + fake_loss) * 0.5


  def compile(
    self,
    gen_g_optimizer=Adam(lr=0.0002, beta_1=0.5),
    gen_f_optimizer=Adam(lr=0.0002, beta_1=0.5),
    disc_x_optimizer=Adam(lr=0.0002, beta_1=0.5),
    disc_y_optimizer=Adam(lr=0.0002, beta_1=0.5)
  ):

    super().compile()
    self.gen_g_optimizer = gen_g_optimizer
    self.gen_f_optimizer = gen_f_optimizer
    self.disc_x_optimizer = disc_x_optimizer
    self.disc_y_optimizer = disc_y_optimizer
    self.generator_loss_fn = self.g_loss_fn
    self.discriminator_loss_fn = self.d_loss_fn
    self.cycle_loss_fn = self.adv_loss_fn
    self.identity_loss_fn = self.identity_loss_fn

  def call(self):
    raise NotImplementedError("Directly call either the Generator or "
        "Discriminator model during inference.")

  def train_step(self, batch_data):
    # real_x is the ailiased data
    # real_y is the clean data
    real_x, real_y = batch_data

    # We need to set `persistent=True` since we need to use
    # the tape to calculate the derivatives for both generators.
    # If it didn't persist, we'd not be able to update both generators
    # since the data would be erased after the first generator has
    # completed its backpropagation pass.
    with tf.GradientTape(persistent=True) as tape:
      # Ailiased to clean
      fake_y = self.gen_g(real_x, training=True)
      # Clean to ailiased
      fake_x = self.gen_f(real_y, training=True)

      # Cycle (Ailiased -> Fake Clean -> Fake Ailiased)
      cycle_x = self.gen_f(fake_y, training=True)
      # Cycle (Clean -> Fake Ailiased -> Fake Clean)
      cycle_y = self.gen_g(fake_x, training=True)

      # Identity mapping
      same_x = self.gen_f(real_x, training=True)
      same_y = self.gen_g(real_y, training=True)

      # Discriminator outputs
      disc_real_x = self.disc_x(real_x, training=True)
      disc_fake_x = self.disc_x(fake_x, training=True)

      disc_real_y = self.disc_y(real_y, training=True)
      disc_fake_y = self.disc_y(fake_y, training=True)

      # Adversarial Loss for Generators
      gen_g_loss = self.generator_loss_fn(self, disc_fake_y)
      gen_f_loss = self.generator_loss_fn(self, disc_fake_x)

      # Cycle Loss for Generators
      cycle_loss_g = self.lambda_cycle * self.cycle_loss_fn(real_y, cycle_y)
      cycle_loss_f = self.lambda_cycle * self.cycle_loss_fn(real_x, cycle_x)

      # Generator identity losses
      id_loss_g = (
        self.identity_loss_fn(real_y, same_y)
        * self.lambda_cycle
        * self.lambda_identity
      )

      id_loss_f = (
        self.identity_loss_fn(real_x, same_x)
        * self.lambda_cycle
        * self.lambda_identity
      )

      # Total generator loss
      total_loss_g = gen_g_loss + cycle_loss_g + id_loss_g
      total_loss_f = gen_f_loss + cycle_loss_f + id_loss_f

      # Discriminator loss
      disc_x_loss = self.discriminator_loss_fn(self, disc_real_x, disc_fake_x)
      disc_y_loss = self.discriminator_loss_fn(self, disc_real_y, disc_fake_y)

    # Get the generator gradients using tape:
    grads_g = tape.gradient(total_loss_g, self.gen_G.trainable_variables)
    grads_f = tape.gradient(total_loss_f, self.gen_F.trainable_variables)

    # Get the discriminator gradients using tape:
    disc_x_grads = tape.gradient(disc_x_loss, self.disc_X.trainable_variables)
    disc_y_grads = tape.gradient(disc_y_loss, self.disc_Y.trainable_variables)

    # Update the weights of the generators
    self.gen_g_optimizer.apply_gradients(
      zip(grads_g, self.gen_g.trainable_variables)
    )
    self.gen_f_optimizer.apply_gradients(
      zip(grads_f, self.gen_f.trainable_variables)
    )

    # Update the weights of the discriminators
    self.disc_x_optimizer.apply_gradients(
      zip(disc_x_grads, self.disc_x.trainable_variables)
    )
    self.disc_y_optimizer.apply_gradients(
      zip(disc_y_grads, self.disc_y.trainable_variables)
    )

    return {
      "g_loss": total_loss_g,
      "f_loss": total_loss_f,
      "d_x_loss": disc_x_loss,
      "d_y_loss": disc_y_loss,
    }
