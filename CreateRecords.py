
"""Converts data to TFRecord.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from collections import namedtuple
from datetime import datetime
from scipy.misc import imresize

import json
import os.path
import random
import sys
import threading

import numpy as np
import pickle as pkl
import scipy.io as sio
import tensorflow as tf

tf.flags.DEFINE_string("out_dir", "output/",
											 "Image directory.")
tf.flags.DEFINE_string("src_dir", "input/",
											 "Directory containing src images.")
tf.flags.DEFINE_string("trg_dir", "target/",
											 "Directory containing masks.")

tf.flags.DEFINE_string("train_label_file", "datalist1.txt",
											 "Training text file.")
tf.flags.DEFINE_string("tf_dir", "tfrecord", "Output data directory.")
tf.flags.DEFINE_string("prefix", "", "Prefix of the tensorflow record files.")

tf.flags.DEFINE_integer("train_shards", 8,
												"Number of shards in training TFRecord files.")
tf.flags.DEFINE_integer("num_threads", 8,
												"Number of threads to preprocess the images.")
tf.flags.DEFINE_string("tps_dir", "control_points/",
						"Directory contains tps transformed images.")

FLAGS = tf.flags.FLAGS

ImageMetadata = namedtuple("ImageMetadata",
													 ["out_id", "src_id", "trg_id"])

class ImageDecoder(object):
	"""Helper class for decoding images in TensorFlow."""

	def __init__(self):
		# Create a single TensorFlow Session for all image decoding calls.
		config = tf.ConfigProto()
		config.gpu_options.visible_device_list= '0,1,2'
		self._sess = tf.Session(config=config)

		# TensorFlow ops for JPEG decoding.
		self._encoded_jpeg = tf.placeholder(dtype=tf.string)
		self._encoded_png = tf.placeholder(dtype=tf.string)
		self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg, channels=3)
		self._decode_png = tf.image.decode_png(self._encoded_png, channels=3)

		self._decoded_jpeg = tf.placeholder(dtype=tf.uint8, shape=[None, None, None])
		self._encode_jpeg = tf.image.encode_jpeg(self._decoded_jpeg)

	def decode_jpeg(self, encoded_jpeg):
		image = self._sess.run(self._decode_jpeg,
													 feed_dict={self._encoded_jpeg: encoded_jpeg})
		assert len(image.shape) == 3
		assert image.shape[2] == 3
		return image

	def decode_png(self, encoded_png):
		image = self._sess.run(self._decode_png,
													 feed_dict={self._encoded_png: encoded_png})
		assert len(image.shape) == 3
		assert image.shape[2] == 3
		return image

	def encode_jpeg(self, decoded_image):
		image = self._sess.run(self._encode_jpeg,
													 feed_dict={self._decoded_jpeg: decoded_image})
		return image

def _int64_feature(value):
	"""Wrapper for inserting an int64 Feature into a SequenceExample proto."""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	"""Wrapper for inserting a bytes Feature into a SequenceExample proto."""
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))

def _float_feature(values):
	return tf.train.Feature(float_list=tf.train.FloatList(value=[a for a in values]))

def _int64_feature_list(values):
	"""Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
	return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
	"""Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
	return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

def _to_tf_example(image, decoder):
	"""Builds a TF Example proto for an image triplet.

	Args:
		image: An ImageMetadata object.
		decoder: An ImageDecoder object.

	Returns:
		A TF Example proto.
	"""
	with open(FLAGS.out_dir + image.out_id, "r") as f:
		encoded_out = f.read()
	with open(FLAGS.src_dir + image.src_id, "r") as f:
		encoded_src = f.read()
	with open(FLAGS.trg_dir + image.trg_id, "r") as f:
		encoded_trg = f.read()
	
	try:
		decoded_image = decoder.decode_jpeg(encoded_out)
		decoder.decode_jpeg(encoded_src)
		
	# try:
	# 	decoder.decode_png(encoded_human)
	# except:
	# 	print("Skipping file with invalid PNG Human data: %s" % image.image_id)
	except (tf.errors.InvalidArgumentError, AssertionError):
		print("Skipping file with invalid JPEG data: %s" % image.image_id)
		print("Skipping file with invalid JPEG data: %s" % image.product_image_id)
		return

	height = decoded_image.shape[0]
	width = decoded_image.shape[1]
	
	tps_name = image.src_id+'_tps.mat'#tps
	if os.path.isfile(FLAGS.tps_dir + tps_name):
		print('TPS file exists')
		tps_image = sio.loadmat(FLAGS.tps_dir + tps_name)
		tps_control_points = tps_image["control_points"].reshape(-1)
	else:
		print("Skipping samples without TPS control points: %s" % image.out_id, tps_name)
		return


	tf_example = tf.train.Example(features=tf.train.Features(feature={
				'out_id': _bytes_feature(image.out_id),
				'src_id': _bytes_feature(image.src_id),
				'trg_id': _bytes_feature(image.trg_id),
				'out_image': _bytes_feature(encoded_out),
				'src_image': _bytes_feature(encoded_src),
				'trg_image': _bytes_feature(encoded_trg),
				'height': _int64_feature(height),
				'width': _int64_feature(width),
				'tps_control_points': _float_feature(tps_control_points),
				}))

	return tf_example


def _process_image_files(thread_index, ranges, name, images, decoder,
														num_shards):
	"""Processes and saves a subset of images as TFRecord files in one thread.

	Args:
		thread_index: Integer thread identifier within [0, len(ranges)].
		ranges: A list of pairs of integers specifying the ranges of the dataset to
			process in parallel.
		name: Unique identifier specifying the dataset.
		images: List of ImageMetadata.
		decoder: An ImageDecoder object.
		num_shards: Integer number of shards for the output files.
	"""
	# Each thread produces N shards where N = num_shards / num_threads. For
	# instance, if num_shards = 128, and num_threads = 2, then the first thread
	# would produce shards [0, 64).
	num_threads = len(ranges)
	assert not num_shards % num_threads
	num_shards_per_batch = int(num_shards / num_threads)

	shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
														 num_shards_per_batch + 1).astype(int)
	num_images_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

	counter = 0
	for s in xrange(num_shards_per_batch):
		# Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
		shard = thread_index * num_shards_per_batch + s
		output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
		output_file = os.path.join(FLAGS.tf_dir, output_filename)
		writer = tf.python_io.TFRecordWriter(output_file)

		shard_counter = 0
		images_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
		for i in images_in_shard:
			image = images[i]

			tf_example = _to_tf_example(image, decoder)
			if tf_example is not None:
				writer.write(tf_example.SerializeToString())
				shard_counter += 1
				counter += 1

			if not counter % 1000:
				print("%s [thread %d]: Processed %d of %d items in thread batch." %
							(datetime.now(), thread_index, counter, num_images_in_thread))
				sys.stdout.flush()

		writer.close()
		print("%s [thread %d]: Wrote %d image pairs to %s" %
					(datetime.now(), thread_index, shard_counter, output_file))
		sys.stdout.flush()
		shard_counter = 0
	print("%s [thread %d]: Wrote %d image pairs to %d shards." %
				(datetime.now(), thread_index, counter, num_shards_per_batch))
	sys.stdout.flush()


def _process_dataset(name, images, num_shards):
	"""Processes a complete data set and saves it as a TFRecord.

	Args:
		name: Unique identifier specifying the dataset.
		images: List of ImageMetadata.
		num_shards: Integer number of shards for the output files.
	"""
	
	# Shuffle the ordering of images. Make the randomization repeatable.
	random.seed(12345)
	random.shuffle(images)

	# Break the images into num_threads batches. Batch i is defined as
	# images[ranges[i][0]:ranges[i][1]].
	num_threads = min(num_shards, FLAGS.num_threads)
	spacing = np.linspace(0, len(images), num_threads + 1).astype(np.int)
	ranges = []
	threads = []
	for i in xrange(len(spacing) - 1):
		ranges.append([spacing[i], spacing[i + 1]])

	# Create a mechanism for monitoring when all threads are finished.
	coord = tf.train.Coordinator()

	# Create a utility for decoding JPEG images to run sanity checks.
	decoder = ImageDecoder()

	# Launch a thread for each batch.
	print("Launching %d threads for spacings: %s" % (num_threads, ranges))
	for thread_index in xrange(len(ranges)):
		args = (thread_index, ranges, name, images, decoder, num_shards)
		t = threading.Thread(target=_process_image_files, args=args)
		t.start()
		threads.append(t)

	# Wait for all the threads to terminate.
	coord.join(threads)
	print("%s: Finished processing all %d image pairs in data set '%s'." %
				(datetime.now(), len(images), name))



def _load_and_process_metadata(label_file):
	"""Loads image metadata from a text file
	Args:
		label_file: txt file containing pair annotations.
		
	Returns:
		A list of ImageMetadata.
	"""
	
	image_pairs = open(label_file).read().splitlines()
	image_metadata = []
	
	for item in image_pairs:
		name1 = item.strip()
		image_pair = [name1, name1[:-5]+'1.jpg_pad.jpg', name1[:-5]+'1.jpg_mask.jpg']
		print(image_pair)
		# print (image_pair)
		image_metadata.append(ImageMetadata(image_pair[0], image_pair[1], image_pair[2]))
		
	print("Finished processing %d pairs for %d images in %s" %
				(len(image_pairs), len(image_pairs), label_file))

	return image_metadata


def main(unused_argv):
	def _is_valid_num_shards(num_shards):
		"""Returns True if num_shards is compatible with FLAGS.num_threads."""
		return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

	assert _is_valid_num_shards(FLAGS.train_shards), (
			"Please make the FLAGS.num_threads commensurate with FLAGS.train_shards")

	if not tf.gfile.IsDirectory(FLAGS.tf_dir):
		tf.gfile.MakeDirs(FLAGS.tf_dir)

	
	# Load image metadata from label files.
	train_dataset = _load_and_process_metadata(FLAGS.train_label_file)
	
	print(len(train_dataset))

	_process_dataset(FLAGS.prefix + "train", train_dataset, FLAGS.train_shards)


if __name__ == "__main__":
	tf.app.run()
