from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import time
import numpy as np
import scipy.io as sio
import tensorflow as tf
from scipy.misc import imresize

input_file_pattern = "./tfrecord/train-?????-of-00008"
values_per_input_shard = 443
num_preprocess_threads = 1
batch_size = 1

def process_image(encoded_out,
									encoded_src,
									encoded_trg,
									is_training=True,
									height=256,
									width=192,
									resize_height=256,
									resize_width=192,
									thread_id=0,
									image_format="jpeg",
									zero_one_mask=True,
									different_image_size=False):
	
	
	def image_summary(name, image):
		return
		
	# Decode image into a float32 Tensor of shape [?, ?, 3] with values in [0, 1).
	with tf.name_scope("decode", values=[encoded_out]):
		if image_format == "jpeg":
			image = tf.image.decode_jpeg(encoded_out, channels=3)
			human = tf.image.decode_jpeg(encoded_src, channels=3)
			prod_image = tf.image.decode_jpeg(encoded_trg, channels=3)
		elif image_format == "png":
			image = tf.image.decode_png(encoded_out, channels=3)
			human = tf.image.decode_png(encoded_src, channels=3)
			prod_image = tf.image.decode_png(encoded_trg, channels=3)
		else:
			raise ValueError("Invalid image format: %s" % image_format)

	image = tf.image.convert_image_dtype(image, dtype=tf.float32)
	human = tf.image.convert_image_dtype(human, dtype=tf.float32)
	prod_image = tf.image.convert_image_dtype(prod_image, dtype=tf.float32)
	image_summary("original_image", image)
	image_summary("original_human", human)
	image_summary("original_prod_image", prod_image)
	

	# Resize image.
	assert (resize_height > 0) == (resize_width > 0)
	if different_image_size:
		image = tf.image.resize_images(image,
																	 size=[height, width],
																	 method=tf.image.ResizeMethod.BILINEAR)
		human = tf.image.resize_images(human,
																	 size=[height, width],
																	 method=tf.image.ResizeMethod.BILINEAR)
		prod_image = tf.image.resize_images(prod_image,
																				size=[height, width],
																				method=tf.image.ResizeMethod.BILINEAR)
	else:
		image = tf.image.resize_images(image,
																	 size=[resize_height, resize_width],
																	 method=tf.image.ResizeMethod.BILINEAR)
		human = tf.image.resize_images(human,
																	 size=[resize_height, resize_width],
																	 method=tf.image.ResizeMethod.BILINEAR)
		prod_image = tf.image.resize_images(prod_image,
																				size=[resize_height, resize_width],
																				method=tf.image.ResizeMethod.BILINEAR)


	
	image_summary("final_image", image)
	image_summary("final_human", human)
	image_summary("final_prod_image", prod_image)
	
	# Rescale to [-1,1] instead of [0, 1]
	# image = (image - 0.5) * 2.0
	# human = (human - 0.5) * 2.0
	# prod_image = (prod_image - 0.5) * 2.0
	
		
	
	# human, green, blue= tf.split(3, 3, human)
	return image, human, prod_image







def parse_tf_example(serialized, stage=""):
	features = tf.parse_single_example(
			serialized,
			features={
					"out_id": tf.FixedLenFeature([], tf.string),
					"src_id": tf.FixedLenFeature([], tf.string),
					"trg_id": tf.FixedLenFeature([], tf.string),
					"out_image": tf.FixedLenFeature([], tf.string),
					"src_image": tf.FixedLenFeature([], tf.string),
					"trg_image": tf.FixedLenFeature([], tf.string),
					"height": tf.FixedLenFeature([], tf.int64),
					"width": tf.FixedLenFeature([], tf.int64),
					"tps_control_points": tf.VarLenFeature(tf.float32),
			}
	)
	encoded_out = features["out_image"]
	encoded_src = features["src_image"]
	encoded_trg = features["trg_image"]

	height = tf.cast(features["height"], tf.int32)
	width = tf.cast(features["width"], tf.int32)

	tps_points = features["tps_control_points"]
	# tps_points = tf.sparse_tensor_to_dense(tps_points, default_value=0.)
	# tps_points = tf.reshape(tps_points, tf.stack([2,10,10]))

	tps_points = tf.reshape(tps_points,[1,100,2])
	tps_points = tf.cast(tps_points, dtype=tf.float32)
	#tps_points = tf.transpose(tf.reshape(tps_points, tf.stack([2, 100]))) * 2 - 1
	
	return (encoded_out, encoded_src, encoded_trg, features["out_id"],features["src_id"],features["trg_id"], tps_points)
def prefetch_input_data(reader,
												file_pattern,
												is_training,
												batch_size,
												values_per_shard,
												input_queue_capacity_factor=16,
												num_reader_threads=1,
												shard_queue_name="filename_queue",
												value_queue_name="input_queue"):
	"""Prefetches string values from disk into an input queue.

	In training the capacity of the queue is important because a larger queue
	means better mixing of training examples between shards. The minimum number of
	values kept in the queue is values_per_shard * input_queue_capacity_factor,
	where input_queue_memory factor should be chosen to trade-off better mixing
	with memory usage.

	Args:
		reader: Instance of tf.ReaderBase.
		file_pattern: Comma-separated list of file patterns (e.g.
				/tmp/train_data-?????-of-00100).
		is_training: Boolean; whether prefetching for training or eval.
		batch_size: Model batch size used to determine queue capacity.
		values_per_shard: Approximate number of values per shard.
		input_queue_capacity_factor: Minimum number of values to keep in the queue
			in multiples of values_per_shard. See comments above.
		num_reader_threads: Number of reader threads to fill the queue.
		shard_queue_name: Name for the shards filename queue.
		value_queue_name: Name for the values input queue.

	Returns:
		A Queue containing prefetched string values.
	"""
	data_files = []
	for pattern in file_pattern.split(","):
		data_files.extend(tf.gfile.Glob(pattern))
	if not data_files:
		tf.logging.fatal("Found no input files matching %s", file_pattern)
	else:
		tf.logging.info("Prefetching values from %d files matching %s",
										len(data_files), file_pattern)

	if is_training:
		filename_queue = tf.train.string_input_producer(
				data_files, shuffle=True, capacity=16, name=shard_queue_name)
		min_queue_examples = values_per_shard * input_queue_capacity_factor
		capacity = min_queue_examples + 100 * batch_size
		values_queue = tf.RandomShuffleQueue(
				capacity=capacity,
				min_after_dequeue=min_queue_examples,
				dtypes=[tf.string],
				name="random_" + value_queue_name)
	else:
		filename_queue = tf.train.string_input_producer(
				data_files, shuffle=False, capacity=1, name=shard_queue_name)
		capacity = values_per_shard + 3 * batch_size
		values_queue = tf.FIFOQueue(
				capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name)

	enqueue_ops = []
	for _ in range(num_reader_threads):
		_, value = reader.read(filename_queue)
		enqueue_ops.append(values_queue.enqueue([value]))
	tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
			values_queue, enqueue_ops))
	tf.summary.scalar(
			"queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
			tf.cast(values_queue.size(), tf.float32) * (1. / capacity))

	return values_queue
def build_input():
	# Load input data
	input_queue = prefetch_input_data(
			tf.TFRecordReader(),
			input_file_pattern,
			is_training=True,
			batch_size=batch_size,
			values_per_shard=values_per_input_shard,
			input_queue_capacity_factor=2,
			num_reader_threads=num_preprocess_threads)

	# Image processing and random distortion. Split across multiple threads
	images_and_maps = []

	for thread_id in range(num_preprocess_threads):
		serialized_example = input_queue.dequeue()
		(encoded_out, encoded_src, encoded_trg, out_id, src_id, trg_id, tps_points) = parse_tf_example(serialized_example)

		#Body Segment is Human now
		(out_image, src_image, trg_image) = process_image(encoded_out, encoded_src, encoded_trg)

		images_and_maps.append([src_image, trg_image,out_image, tps_points])

	# Batch inputs.
	queue_capacity = (7 * num_preprocess_threads *
										batch_size)

	return tf.train.batch_join(images_and_maps,
														 batch_size=batch_size,
														 capacity=queue_capacity,
														 name="batch")

def main():
	(src_image, trg_image,out_image, tps_points)=build_input()
	print(out_image.shape, src_image.shape, trg_image.shape, tps_points.shape)
	coord = tf.train.Coordinator()
	sess = tf.Session()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	m = sess.run(out_image)
	print (np.unique(m))
	
if __name__ == "__main__":
	main()
	#tf.app.run()