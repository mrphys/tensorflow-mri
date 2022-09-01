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
# limitations under the License.
# ==============================================================================

"""CycleGAN Layers and Composite Generator/Discriminator Model Implementation."""

import tensorflow as tf
from keras import layers
import tensorflow_addons as tfa

from keras.optimizers import Adam
from keras.initializers import RandomNormal

from keras.models import Model
from keras.layers import Layer
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from tensorflow_addons.layers import InstanceNormalization

from keras.activations import sigmoid
from keras.losses import MeanSquaredError, MeanAbsoluteError

class CGResNetBlock(Layer):
	"""ResNet Block Layer specific to the CycleGAN Generator Architecture."""
	def __init__(self, n_filters):
		"""
		Args:
			n_filters: An `int` denoting the number of convolutional filters
			to use, and same dtype as `self`.
		"""
		super(CGResNetBlock, self).__init__()
		self.init = RandomNormal(stddev=0.02)
		self.conv2a = Conv2D(n_filters, (3,3), padding="same", kernel_initializer=self.init)
		self.conv2b = Conv2D(256, (3,3), padding="same", kernel_initializer=self.init)
		self.instance_norm1 = InstanceNormalization(axis=-1)
		self.instance_norm2 = InstanceNormalization(axis=-1)
		self.relu = Activation("relu")
		self.concat = Concatenate()

	def call(self, input_tensor):
		x = self.conv2a(input_tensor)
		x = self.instance_norm1(x)
		x = self.relu(x)
		x = self.conv2b(x)
		x = self.instance_norm2(x)
		x = self.concat([x, input_tensor])
		return x

class CGEncoder(Layer):
	"""Encoder Layer specific to the CycleGAN Generator Architecture."""

	def __init__(self):
		super(CGEncoder, self).__init__()
		self.init = RandomNormal(stddev=0.02)
		self.conv2a = Conv2D(64, (7,7), padding='same', kernel_initializer=self.init)
		self.conv2b = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=self.init)
		self.conv2c = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=self.init)
		self.instance_norm1 = InstanceNormalization(axis=-1)
		self.instance_norm2 = InstanceNormalization(axis=-1)
		self.instance_norm3 = InstanceNormalization(axis=-1)
		self.relu = Activation('relu')

	def call(self, input_tensor):
		x = self.conv2a(input_tensor)
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
		super(CGDecoder, self).__init__()
		self.init = RandomNormal(stddev=0.02)
		self.conv2ta = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=self.init)
		self.conv2tb = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=self.init)
		self.conv2c = Conv2D(1, (7,7), padding='same', kernel_initializer=self.init)
		self.instance_norm1 = InstanceNormalization(axis=-1)
		self.instance_norm2 = InstanceNormalization(axis=-1)
		self.instance_norm3 = InstanceNormalization(axis=-1)
		self.tanh = Activation('tanh')
		self.relu = Activation('relu')

	def call(self, input_tensor):
		x = self.conv2ta(input_tensor)
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
	"""Convolutional Encoder Layer specific to the CycleGAN Discriminator Architecture."""
	def __init__(self):
		super(CGConvEncoder, self).__init__()
		self.init = RandomNormal(stddev=0.02)

		self.conv2a = Conv2D(64, (4,4), strides=(3,3), padding='same', kernel_initializer=self.init)
		self.conv2b = Conv2D(128, (4,4), strides=(3,3), padding='same', kernel_initializer=self.init)
		self.conv2c = Conv2D(256, (4,4), strides=(3,3), padding='same', kernel_initializer=self.init)
		self.conv2d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=self.init)
		self.conv2e = Conv2D(512, (4,4), padding='same', kernel_initializer=self.init)
		self.conv2f = Conv2D(1, (4,4), padding='same', kernel_initializer=self.init)
		
		self.instance_norm1 = InstanceNormalization(axis=-1)
		self.instance_norm2 = InstanceNormalization(axis=-1)
		self.instance_norm3 = InstanceNormalization(axis=-1)
		self.instance_norm4 = InstanceNormalization(axis=-1)
		
		self.leakyReLU = LeakyReLU(alpha=0.2)
		self.sigmoid = sigmoid
	
	def call(self, input_tensor):
		"""
		Args:
			input_tensor: A `tf.Tensor` of shape `[..., n]` and same dtype as `self`.
		Returns:
			A `tf.Tensor` of shape `[..., 1]` and same dtype as `self`.
		"""
		x = self.conv2a(input_tensor)
		x = self.leakyReLU(x)
		
		x = self.conv2b(x)
		x = self.instance_norm1(x)
		x = self.leakyReLU(x)
		
		x = self.conv2c(x)
		x = self.instance_norm2(x)
		x = self.leakyReLU(x)
		
		x = self.conv2d(x)
		x = self.instance_norm3(x)
		x = self.leakyReLU(x)
		
		x = self.conv2e(x)
		x = self.instance_norm4(x)
		x = self.leakyReLU(x)

		out_pred = self.conv2f(x)

		return out_pred

class CGGenerator(Model):
	"""CycleGAN Generator Model."""
	def __init__(self, image_shape, n_resnet=None):
		super(CGGenerator, self).__init__()

		# If image size is less than 256x256, only use
		# 6 resnet blocks
		if image_shape[1] < 256 and n_resnet == None:
			pass
	
		self.encoder = CGEncoder()
		self.decoder = CGDecoder()
		self.res_blocks = []
	
	def call(self, inputs):
		x = self.encoder(inputs)
		for i in range(0, self.n_resnet):
			self.res_blocks.append(CGResNetBlock(256))
			x = self.res_blocks[i](x)
		x = self.decoder(x)
		return x

class CGDiscriminator(Model):
	"""CycleGAN Discriminator Model."""
	def __init__(self):
		super(CGDiscriminator, self).__init__()
		self.conv_encoder = CGConvEncoder()

	def compile(self, optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5], **kwargs):
		super().compile(optimizer=optimizer, losds_weights=loss_weights **kwargs)

	def call(self, input_image):
		return self.conv_encoder(input_image) 

class CycleGAN(Model):
	"""
	This is the main composite CycleGAN  Model used to concurrently train both generators
	and discriminators with a custom training loop.
	
	Adapted from the following: https://keras.io/examples/generative/cyclegan/
	
	By default, the image size for both Generators and Discriminators
	is set to be 128x128 - but these can be changed to any square image
	dimension, although > 256x256 and < 64x64 is not recommended.
	"""

	def __init__(
		self,
		generator_G = CGGenerator(image_shape=(1,128,128)),
		generator_F = CGGenerator(image_shape=(1,128,128)),
		discriminator_X = CGDiscriminator(),
		discriminator_Y = CGDiscriminator(),
		lambda_cycle=10.0,
		lambda_identity=1.0,
	):
		super(CycleGAN, self).__init__()
		self.gen_G = generator_G
		self.gen_F = generator_F
		self.disc_X = discriminator_X
		self.disc_Y = discriminator_Y
		self.lambda_cycle = lambda_cycle
		self.lambda_identity = lambda_identity
	
		# Loss function for evaluating adversarial loss
		self.adv_loss_fn = MeanSquaredError()

	# Define the loss function for the generators
	def generator_loss_fn(self, fake):
		fake_loss = self.adv_loss_fn(tf.ones_like(fake), fake)
		return fake_loss

	# Define the loss function for the discriminators
	def discriminator_loss_fn(self, real, fake):
		real_loss = self.adv_loss_fn(tf.ones_like(real), real)
		fake_loss = self.adv_loss_fn(tf.zeros_like(fake), fake)
		return (real_loss + fake_loss) * 0.5
	
	def compile(
		self,
		gen_G_optimizer=Adam(lr=0.0002, beta_1=0.5),
		gen_F_optimizer=Adam(lr=0.0002, beta_1=0.5),
		disc_X_optimizer=Adam(lr=0.0002, beta_1=0.5),
		disc_Y_optimizer=Adam(lr=0.0002, beta_1=0.5),
		gen_loss_fn=generator_loss_fn,
		disc_loss_fn=discriminator_loss_fn,
	):

		super(CycleGAN, self).compile()
		self.gen_G_optimizer = gen_G_optimizer
		self.gen_F_optimizer = gen_F_optimizer
		self.disc_X_optimizer = disc_X_optimizer
		self.disc_Y_optimizer = disc_Y_optimizer
		self.generator_loss_fn = gen_loss_fn
		self.discriminator_loss_fn = disc_loss_fn
		self.cycle_loss_fn = MeanAbsoluteError()
		self.identity_loss_fn = MeanAbsoluteError()
	
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
			fake_y = self.gen_G(real_x, training=True)
			# Clean to ailiased
			fake_x = self.gen_F(real_y, training=True)

			# Cycle (Ailiased -> Fake Clean -> Fake Ailiased)
			cycle_x = self.gen_F(fake_y, training=True)
			# Cycle (Clean -> Fake Ailiased -> Fake Clean)
			cycle_y = self.gen_G(fake_x, training=True)

			# Identity mapping 
			same_x = self.gen_F(real_x, training=True)
			same_y = self.gen_G(real_y, training=True)

			# Discriminator outputs
			disc_real_x = self.disc_X(real_x, training=True)
			disc_fake_x = self.disc_X(fake_x, training=True)

			disc_real_y = self.disc_Y(real_y, training=True)
			disc_fake_y = self.disc_Y(fake_y, training=True)

			# Adversarial Loss for Generators
			gen_G_loss = self.generator_loss_fn(self, disc_fake_y)
			gen_F_loss = self.generator_loss_fn(self, disc_fake_x)
			
			# Cycle Loss for Generators
			cycle_loss_G = self.lambda_cycle * self.cycle_loss_fn(real_y, cycle_y)
			cycle_loss_F = self.lambda_cycle * self.cycle_loss_fn(real_x, cycle_x)
			
			# Generator identity losses
			id_loss_G = (
				self.identity_loss_fn(real_y, same_y)
				* self.lambda_cycle
				* self.lambda_identity
			)

			id_loss_F = (
				self.identity_loss_fn(real_x, same_x)
				* self.lambda_cycle
				* self.lambda_identity
			)

			# Total generator loss
			total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
			total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F

			# Discriminator loss
			disc_X_loss = self.discriminator_loss_fn(self, disc_real_x, disc_fake_x)
			disc_Y_loss = self.discriminator_loss_fn(self, disc_real_y, disc_fake_y)
		
		# Get the generator gradients using tape:
		grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
		grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)

		# Get the discriminator gradients using tape:
		disc_X_grads = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
		disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

		# Update the weights of the generators
		self.gen_G_optimizer.apply_gradients(
			zip(grads_G, self.gen_G.trainable_variables)
		)
		self.gen_F_optimizer.apply_gradients(
			zip(grads_F, self.gen_F.trainable_variables)
		)

		# Update the weights of the discriminators
		self.disc_X_optimizer.apply_gradients(
			zip(disc_X_grads, self.disc_X.trainable_variables)
		)
		self.disc_Y_optimizer.apply_gradients(
			zip(disc_Y_grads, self.disc_Y.trainable_variables)
		)

		return {
			"G_loss": total_loss_G,
			"F_loss": total_loss_F,
			"D_X_loss": disc_X_loss,
			"D_Y_loss": disc_Y_loss,
		}