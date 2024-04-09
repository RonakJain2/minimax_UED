"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import chex
from tensorflow_probability.substrates import jax as tfp

from minimax.models import common
from minimax.models import s5
from minimax.models.registration import register


class MountainCarModel(nn.Module):
	"""Split Actor-Critic Architecture for PPO."""
	output_dim: int = 3
	n_hidden_layers: int = 1
	hidden_dim: int = 32

	n_conv_filters: int = 0 # changed for mountain car
	conv_kernel_size: int = 3
	n_scalar_embeddings: int = 0 # changed for mountain car
	max_scalar: int = 4
	scalar_embed_dim: int = 0 # changed for mountain car
	recurrent_arch: str = None
	recurrent_hidden_dim: int = 256
	base_activation: str = 'relu'
	head_activation: str = 'tanh'

	s5_n_blocks: int = 2
	s5_n_layers: int = 4
	s5_layernorm_pos: str = None
	s5_activation: str = "half_glu1"

	value_ensemble_size: int = 1

	def setup(self):
		self.conv = nn.Sequential([
			nn.Conv(
				features=self.n_conv_filters, 
				kernel_size=[self.conv_kernel_size,]*2,
				strides=1, 
				kernel_init=common.init_orth(
					scale=common.calc_gain(self.base_activation)
				),
				padding='VALID',
				name='cnn'),
			common.get_activation(self.base_activation)
		])

		if self.n_scalar_embeddings > 0:
			self.fc_scalar = nn.Embed(
				num_embeddings=self.n_scalar_embeddings,
				features=self.scalar_embed_dim, 
				embedding_init=common.init_orth(
					common.calc_gain('linear')
				),
				name=f'fc_scalar'
			)
		elif self.scalar_embed_dim > 0:
			self.fc_scalar = nn.Dense(
				self.scalar_embed_dim,
				kernel_init=common.init_orth(
					common.calc_gain('linear')
				),
				name=f'fc_scalar'
			)
		else:
			self.fc_scalar = None

		if self.recurrent_arch is not None:
			if self.recurrent_arch == 's5':
				self.embed_pre_s5 = nn.Sequential([
					nn.Dense(
						self.recurrent_hidden_dim,
						kernel_init=common.init_orth(
							common.calc_gain('linear')
						),
						name=f'fc_pre_s5'
						)
				])
				self.rnn = s5.make_s5_encoder_stack(
					input_dim=self.recurrent_hidden_dim,
					ssm_state_dim=self.recurrent_hidden_dim,
					n_blocks=self.s5_n_blocks,
					n_layers=self.s5_n_layers,
					activation=self.s5_activation,
					layernorm_pos=self.s5_layernorm_pos
				)
			else:
				self.rnn = common.ScannedRNN(
					recurrent_arch=self.recurrent_arch,
					recurrent_hidden_dim=self.recurrent_hidden_dim,
					kernel_init=common.init_orth(),
					recurrent_kernel_init=common.init_orth()
				)
		else:
			self.rnn = None

		self.pi_head = nn.Sequential([
			common.make_fc_layers(
				'fc_pi', 
				n_layers=self.n_hidden_layers,
				hidden_dim=self.hidden_dim,
				activation=common.get_activation(self.head_activation),
				kernel_init=common.init_orth(
					common.calc_gain(self.head_activation)
				)
			),
			nn.Dense(
				self.output_dim, 
				kernel_init=nn.initializers.constant(0.01), 
				name=f'fc_pi_final'
			)
		])

		value_head_kwargs = dict(
			n_hidden_layers=self.n_hidden_layers, 
			hidden_dim=self.hidden_dim,
			activation=nn.tanh,
			kernel_init=common.init_orth(
				common.calc_gain('tanh')
			),
			last_layer_kernel_init=common.init_orth(
				common.calc_gain('linear')
			)
		)

		if self.value_ensemble_size > 1:
			self.v_head = common.EnsembleValueHead(
				n=self.value_ensemble_size, **value_head_kwargs)
		else:
			self.v_head = common.ValueHead(**value_head_kwargs)

	def __call__(self, x, carry=None):
		raise NotImplementedError

	def initialize_carry(
			self, 
			rng: chex.PRNGKey, 
			batch_dims: Tuple[int] = ()) -> Tuple[chex.ArrayTree, chex.ArrayTree]:
		"""Initialize hidden state of LSTM."""
		if self.recurrent_arch is not None:
			if self.recurrent_arch == 's5':
				return s5.S5EncoderStack.initialize_carry( # Since conj_sym=True
					rng, batch_dims, self.recurrent_hidden_dim//2, self.s5_n_layers
				)
			else:
				return common.ScannedRNN.initialize_carry(
					rng, batch_dims, self.recurrent_hidden_dim, self.recurrent_arch)
		else:
			raise ValueError('Model is not recurrent.')

	@property
	def is_recurrent(self):
		return self.recurrent_arch is not None


class MountainCarStudentModel(MountainCarModel):
	def __call__(self, x, carry=None, reset=None):
		"""
		Inputs:
			x: B x h x w observations
			hxs: B x hx_dim hidden states
			masks: B length vector of done masks
		"""
			
		if self.rnn is not None:
			if self.recurrent_arch == 's5':
				x = self.embed_pre_s5(x)
				carry, x = self.rnn(carry, x, reset)
			else:
				carry, x = self.rnn(carry, (x, reset))

		logits = self.pi_head(x)

		v = self.v_head(x)

		return v, logits, carry


# Register models
if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register(
	env_group_id='mountaincar', model_id='default_student_cnn', 
	entry_point=module_path + ':MountainCarStudentModel')

