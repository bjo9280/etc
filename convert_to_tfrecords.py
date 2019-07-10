import math
import os
import random
import sys
import re
import cv2
import imghdr
import numpy as np

import tensorflow as tf

_slimpath = os.path.join('../', 'slim')
sys.path.insert(0, _slimpath)

from datasets import dataset_utils

_RANDOM_SEED = 4223585
_NUM_SHARDS = 5


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = '%s-%05d-of-%05d.tfrecord' % (
        split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, dataset_dir):
    """Converts the given filenames to a TFRecord dataset.
    Args:
      split_name: The name of the dataset, either 'train' or 'validation'.
      filenames: A list of absolute paths to png or jpg images.
      class_names_to_ids: A dictionary from class names (strings) to ids
        (integers).
      dataset_dir: The directory where the converted datasets are stored.
    """
    assert split_name in ['train', 'validation']

    num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:

            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_dataset_filename(
                    dataset_dir, split_name, shard_id)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))

                    for i in range(start_ndx, end_ndx):
                        image_filename, class_id = filenames[i]
                        print(image_filename, class_id)
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                            i + 1, len(filenames), shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        image_data = tf.gfile.GFile(image_filename, 'rb').read()
                        height, width = image_reader.read_image_dims(sess, image_data)

                        example = dataset_utils.image_to_tfexample(
                            image_data, b'jpg', height, width, int(class_id))
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()

def _dataset_exists(dataset_dir):
    for split_name in ['train', 'validation']:
        for shard_id in range(_NUM_SHARDS):
            output_filename = _get_dataset_filename(
                dataset_dir, split_name, shard_id)
            if not tf.gfile.Exists(output_filename):
                return False
    return True


def txt_to_imglist(txt_file):
    lines = []
    img_list = []
    label_list = []
    with open(txt_file, encoding='utf-8-sig') as f:
        for line in f:
            lines.append([n.replace('\\', '/') for n in line.strip().split(' ')])
        for pair in lines:
            img_list.append((pair[0], pair[1]))
    #             label_list.append()
    return img_list


def run(dataset_dir):
    train_list = txt_to_imglist('imageSets/Train_cls.txt')
    val_list = txt_to_imglist('imageSets/Val_cls.txt')

    # Divide into train and test:
    random.seed(_RANDOM_SEED)
    random.shuffle(train_list)
    random.shuffle(val_list)

    print(len(train_list))
    print(len(val_list))

    # First, convert the training and validation sets.
    _convert_dataset('train', train_list, 'tfrecords')
    _convert_dataset('validation', val_list, 'tfrecords')

if __name__ == "__main__":
    run('exam')
