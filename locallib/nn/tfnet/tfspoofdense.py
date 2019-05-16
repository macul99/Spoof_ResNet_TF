import numpy as np
import tensorflow as tf
from tensorflow.layers import dense, dropout, batch_normalization
from tensorflow.keras.layers import InputLayer
import collections

#from keras.models import Model
#from keras.utils.generic_utils import CustomObjectScope

class DenseBlock(collections.namedtuple('DenseBlock', ['scope', 'unit_fn', 'args'])):
    """A named tuple describing a Dense block.
    """

class SpoofDenseNet:
	@staticmethod
	def block(inputs, n_units, training, act, reg, init, bn_momentum=0.9, drop_rate=0.5, scope=None):
		with tf.variable_scope(scope):
			dense_o = dense(inputs=inputs, units=n_units, activation=act, kernel_regularizer=reg, kernel_initializer=init)
			bn_o = batch_normalization(inputs=dense_o, momentum=bn_momentum, training=training)
			drop_o = dropout(inputs=bn_o, rate=drop_rate, training=training)
		return drop_o

	@staticmethod
	def block_backup(inputs, n_units, training, act, reg, init, bn_momentum=0.9, drop_rate=0.5, scope=None):
		with tf.variable_scope(scope):
			dense_o = dense(inputs=inputs, units=n_units, activation=act, kernel_regularizer=reg, kernel_initializer=init)
			bn_o = batch_normalization(inputs=dense_o, momentum=bn_momentum, training=training) # gamma_initializer=tf.random_normal_initializer(mean=1.0, stddev=0.002),
			drop_o = dropout(inputs=bn_o, rate=drop_rate, training=training)
		return drop_o

	# one for each feature
	@staticmethod
	def build_subnet(inputs, layer_units, training, act, reg, init, args={}, scope=None):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		flag = ('bn_momentum' in args.keys()) and ('drop_rate' in args.keys())

		with tf.variable_scope(scope):
			for i, v in enumerate(layer_units):
				if flag:
					inputs = SpoofDenseNet.block(inputs, v, training, act, reg, init, bn_momentum=args['bn_momentum'], 
													drop_rate=args['drop_rate'], scope="block_%d" % (i+1))
				else:
					inputs = SpoofDenseNet.block(inputs, v, training, act, reg, init, scope="block_%d" % (i+1))

		return inputs

	@staticmethod
	def build_backup(inputs, feature_units, stem_units, training, act, reg, init, args={}, scope=None):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		assert len(inputs) == len(feature_units)

		flag = ('bn_momentum' in args.keys()) and ('drop_rate' in args.keys())

		with tf.variable_scope(scope):
			intermediate_out = []
			for i, sub_in in enumerate(inputs):
				intermediate_out += [SpoofDenseNet.build_subnet(sub_in, feature_units[i], training, act, reg, init, args=args, scope="subnet_%d" % (i+1))]

			net_out = tf.concat(intermediate_out, axis=1, name='intm_concat')

			for i, v in enumerate(stem_units):
				if flag:
					net_out = SpoofDenseNet.block(net_out, v, training, act, reg, init, bn_momentum=args['bn_momentum'], 
													drop_rate=args['drop_rate'], scope="stem_%d" % (i+1))
				else:
					net_out = SpoofDenseNet.block(net_out, v, training, act, reg, init, scope="stem_%d" % (i+1))

		return net_out

	@staticmethod
	def build(inputs, feature_units, stem_units, training, act, reg, init, args={}, scope=None):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		assert len(inputs) == len(feature_units)

		flag = ('bn_momentum' in args.keys()) and ('drop_rate' in args.keys())

		with tf.variable_scope(scope):
			intermediate_out = []
			for i, sub_in in enumerate(inputs):
				intermediate_out += [SpoofDenseNet.build_subnet(sub_in, feature_units[i], training, act, reg, init, args=args, scope="subnet_%d" % (i+1))]

			net_out = tf.concat(intermediate_out, axis=1, name='intm_concat')

		return net_out

	@staticmethod
	def build_test1(inputs, feature_units, stem_units, training, act, reg, init, args={}, scope=None):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		assert len(inputs) == len(feature_units)

		flag = ('bn_momentum' in args.keys()) and ('drop_rate' in args.keys())

		with tf.variable_scope(scope):
			#dense_o = dense(inputs=inputs[0], units=16, activation=act, kernel_regularizer=reg, kernel_initializer=init)
			bn_o = batch_normalization(inputs=inputs[0], momentum=0.9, training=training) # gamma_initializer=tf.random_normal_initializer(mean=1.0, stddev=0.002),
			#drop_o = dropout(inputs=bn_o, rate=0.5, training=training)
		return bn_o