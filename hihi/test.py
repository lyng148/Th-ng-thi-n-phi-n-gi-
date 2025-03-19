import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

# a = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=tf.float32)
# print(a.shape)
# print(a)
# b = tf.nn.softmax(a, axis=-2)
# print(b)

dk = tf.cast(6, dtype=tf.float32)
print(dk)