
# import modules we will use later
from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn import linear_model
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

import time # for random seed


url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None

# hook to report the progress of a download
def download_progress_hook(count, blockSize, totalSize):
	
	global last_percent_reported
	percent = int(count * blockSize * 100 / totalSize)

	if last_percent_reported != percent:
		if percent % 5 == 0:
			sys.stdout.write("%s%%" % percent)
			sys.stdout.flush()
		else:
			sys.stdout.write(".")
			sys.stdout.flush()
	
	last_percent_reported = percent


# download a file if not precent, and make sure it is the right size
# TODO figure out how to use a checksum or something to download this more safely
def maybe_download(filename, force=False):

	# download the file
	if force or not os.path.exists(filename):
		print('Attempting to download:', filename)
		filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
		print('\nDownload Complete!')

	# TODO verify the file downloaded properly
	return filename


# download the files we will use for training
train_filename = maybe_download('notMNIST_large.tar.gz')
test_filename = maybe_download('notMNIST_small.tar.gz')


num_classes = 10
np.random.seed(int(time.time()))


def maybe_extract(filename, force=False):
	root = os.path.splitext(os.path.splitext(filename)[0])[0] # remove .tar.gz

	if os.path.isdir(root) and not force:
		print('%s already present - Skipping extraction.' % root)
	else:
		print('Extracting %s. This may take a while.' % root)
		tar = tarfile.open(filename)
		sys.stdout.flush()
		tar.extractall()
		tar.close()
	
	data_folders = [
		os.path.join(root, d) for d in sorted(os.listdir(root))
		if os.path.isdir(os.path.join(root, d))]
	
	if len(data_folders) != num_classes:
		raise Exception('Expected %d folders, one per class. Found %d instead.' % (num_classes, len(data_folders)))
	
	print(data_folders)
	return data_folders


# Extract the folders used for training
train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)


# display a few sample images
# TODO display sample images properly
dir_path = os.path.dirname(os.path.realpath(__file__))
large_path = dir_path + '/notMNIST_large'
small_path = dir_path + '/notMNIST_small'



# load images
image_size = 28 # images are 28 pixels square
pixel_depth = 255.0 # number of levels per pixel.


# load the data for a single letter label
def load_letter(folder, min_num_images):
	image_files = os.listdir(folder)
	print(len(image_files))
	dataset = np.ndarray(shape=(len(image_files), image_size, image_size), dtype=np.float32)
	print(folder)
	num_images = 0

	for image in image_files:
		image_file = os.path.join(folder, image)
		try:
			image_data = (ndimage.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth
			if image_data.shape != (image_size, image_size):
				raise Exception('Unexpected image shape: %s' % str(image_data.shape))
			dataset[num_images, :, :] = image_data
			num_images = num_images + 1
		except IOError as e:
			print('Could not read:', image_file, ':', e, '- it\s ok, skipping.')

	dataset = dataset[0:num_images, :, :]

	if num_images < min_num_images:
		raise Exception('Many fewer images than expected: %d < %d' % (num_images, min_num_images))

	print('Full dataset tensor:', dataset.shape)
	print('Mean:', np.mean(dataset))
	print('Standard deviation:', np.std(dataset))
	return dataset


# maybe pickle data into desired training folders
def maybe_pickle(data_folders, num_images_per_set, force=False):
	dataset_names = []

	print(data_folders)


	for folder in data_folders:
		set_filename = folder + '.pickle'
		dataset_names.append(set_filename)
		if os.path.exists(set_filename) and not force:
			print ('%s already present - Skipping pickling.' % set_filename)
		else:
			print ('Pickling %s.' % set_filename)
			dataset = load_letter(folder, num_images_per_set)
			try:
				with open(set_filename, 'wb') as f:
					pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
			except Exception as e:
				print('Unable to save data to', set_filename, ':', e)

	return dataset_names


train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)


# display a sample of images and labels
# TODO display images and labels

# check that the data is balanced across classes.
# TODO check that the data is balanced.


# initialize empty arrays for storing dataset
def make_arrays(nb_rows, img_size):
	if nb_rows:
		dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
		labels = np.ndarray(nb_rows, dtype=np.int32)
	else:
		dataset, labels = None, None
	return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
	num_classes = len(pickle_files)
	valid_dataset, valid_labels = make_arrays(valid_size, image_size)
	train_dataset, train_labels = make_arrays(train_size, image_size)
	vsize_per_class = valid_size // num_classes
	tsize_per_class = train_size // num_classes

	start_v, start_t = 0, 0
	end_v, end_t = vsize_per_class, tsize_per_class
	end_l = vsize_per_class + tsize_per_class

	for label, pickle_file in enumerate(pickle_files):
		try:
			with open(pickle_file, 'rb') as f:
				letter_set = pickle.load(f)
				# shuffle leters to have random validation and training set
				np.random.shuffle(letter_set)
				if valid_dataset is not None:
					valid_letter = letter_set[:vsize_per_class, :, :]
					valid_dataset[start_v:end_v, :, :] = valid_letter
					valid_labels[start_v:end_v] = label
					start_v += vsize_per_class
					end_v += vsize_per_class

				train_letter = letter_set[vsize_per_class:end_l, :, :]
				train_dataset[start_t:end_t, :, :] = train_letter
				train_labels[start_t:end_t] = label
				start_t += tsize_per_class
				end_t += tsize_per_class

		except Exception as e:
			print('Unable to process data from', pickle_file, ':', e)
			raise

	return valid_dataset, valid_labels, train_dataset, train_labels


train_size = 200000
valid_size = 10000
test_size = 15000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)


def randomize(dataset, labels):
	permutation = np.random.permutation(labels.shape[0])
	shuffled_dataset = dataset[permutation,:,:]
	shuffled_labels = labels[permutation]
	return shuffled_dataset, shuffled_labels

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

# check data again
# TODO ensure the data is still good after shuffling


pickle_file = 'notMNIST.pickle'

try:
	f = open(pickle_file, 'wb')
	save = {
		'train_dataset': train_dataset,
		'train_labels': train_labels,
		'valid_dataset': valid_dataset,
		'valid_labels': valid_labels,
		'test_dataset': test_dataset,
		'test_labels': test_labels,
	}
	pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
	f.close()
except Exception as e:
	print ('Unable to save data to', pickle_file, ':', e)
	raise

statinfo = os.stat(pickle_file)
print('Compreseed pickle size:', statinfo.st_size)



# measure overlap between training, validation, and test sets.
# TODO measure overlap





# train data on simple model
regression = linear_model.LinearRegression()

# flatten datasets to 2d array
nsamples, nx, ny = train_dataset.shape
d2_train_dataset = train_dataset.reshape((nsamples,nx*ny))

nsamples, nx, ny = valid_dataset.shape
d2_valid_dataset = valid_dataset.reshape((nsamples,nx*ny))

# perform regression fit
regression.fit(d2_train_dataset, train_labels)

print('regression training score: ', regression.score(d2_train_dataset, train_labels))
print('regression validation score: ', regression.score(d2_valid_dataset, valid_labels))


