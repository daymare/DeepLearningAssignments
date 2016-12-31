
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle


# open pickle file from assignment 1
pickle_file = '../1_assignment/notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)


# reformat the data into a more adaptable shape
image_size = 28
num_labels = 10

def reformat(dataset, labels):
	dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
	# Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
	labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
	return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


train_subset = 10000




# define accuracy calculation
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])




# create deep network with l2 regularization
batch_size = 128

graph = tf.Graph()
with graph.as_default():
	# Input data. For the training data, we use a placeholder that will be fed
	# at run time with a training minibatch.
	tf_train_dataset = tf.placeholder(tf.float32,
									shape=(batch_size, image_size * image_size))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)

	# hyperparameters
	dropout_probability = 0.7
	h1_size = 1024
	h2_size = 512
	out_size = 10
	beta = 0.00001

	# create hidden layer 1
	h1_weights = tf.get_variable("h1_weights", shape=[image_size*image_size, h1_size], initializer=tf.contrib.layers.xavier_initializer())
	h1_biases = tf.get_variable("h1_biases", shape=[h1_size])

	# create hidden layer 2
	h2_weights = tf.get_variable("h2_weights", shape=[h1_size, h2_size], initializer=tf.contrib.layers.xavier_initializer())
	h2_biases = tf.get_variable("h2_biases", shape=[h2_size])

	# create output layer
	out_weights = tf.get_variable("out_weights", shape=[h2_size, out_size], initializer=tf.contrib.layers.xavier_initializer())
	out_biases = tf.get_variable("out_biases", shape=[out_size])

	# output function
	def graphOutput(dataset, training=False):
		h1_layer = tf.matmul(dataset, h1_weights) + h1_biases
		if (training == True): h1_layer = tf.nn.dropout(h1_layer, dropout_probability)
		h1_layer = tf.nn.relu(h1_layer)

		h2_layer = tf.matmul(h1_layer, h2_weights) + h2_biases
		if (training == True): h2_layer = tf.nn.dropout(h2_layer, dropout_probability)
		h2_layer = tf.nn.relu(h2_layer)

		out_layer = tf.matmul(h2_layer, out_weights) + out_biases
		return out_layer

	# training computation
	logits = graphOutput(tf_train_dataset, True)

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + tf.nn.l2_loss(h1_weights) * beta + tf.nn.l2_loss(h2_weights) * beta + tf.nn.l2_loss(out_weights) * beta

	# setup optimizer
	optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

	# predictions for datasets
	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(graphOutput(tf_valid_dataset))
	test_prediction = tf.nn.softmax(graphOutput(tf_test_dataset))


# run computation graph
num_steps = 3001

with tf.Session(graph=graph) as session:
	tf.global_variables_initializer().run()
	print("Initialized")
	for step in range(num_steps):
		# Pick an offset within the training data, which has been randomized.
		# Note: we could use better randomization across epochs.
		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
		# Generate a minibatch.
		batch_data = train_dataset[offset:(offset + batch_size), :]
		batch_labels = train_labels[offset:(offset + batch_size), :]
		# Prepare a dictionary telling the session where to feed the minibatch.
		# The key of the dictionary is the placeholder node of the graph to be fed,
		# and the value is the numpy array to feed to it.
		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
		_, l, predictions = session.run(
		  [optimizer, loss, train_prediction], feed_dict=feed_dict)

		if step % 100 == 0: print('step: ', step)

		if (step % 500 == 0):
		  print("Minibatch loss at step %d: %f" % (step, l))
		  print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
		  print("Validation accuracy: %.1f%%" % accuracy(
			valid_prediction.eval(), valid_labels))
	print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


