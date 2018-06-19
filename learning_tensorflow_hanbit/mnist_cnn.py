from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

from layers import conv_layer, max_pool_2x2, full_layer

DATA_DIR = 'tmp/data'
MINBATCH_SIZE = 50
STEPS = 5000

mnist = input_data.read_data_sets(DATA_DIR,one_hot=True)

x = tf.placeholder(tf.float32, shape = [None,784])
y = tf.placeholder(tf.float32, shape = [None,10])
# x 는784의 크기를 가진 텐서를 받을 것이고, y는 정답값이 10개이다

x_image = tf.reshape(x,[-1,28,28,1])
# image를 28 * 28 * 1 의 크기로 재 구성해준다.
# 위의 텐서의 784는 28 * 28이 맞으나 배열처럼 한줄로 펼쳐 놓은것이기 때문에
# 윈도우를 사용하기 위해 2차원 형태로 재구성 해줌 

# convolution layer
conv1 = conv_layer(x_image, shape=[5,5,1,32])
conv1_pool = max_pool_2x2(conv1)

conv2 = conv_layer(conv1_pool, shape= [5,5,32,64])
conv2_pool = max_pool_2x2(conv2)
# 두번 연속 합성곱및 풀링을 통해 계층을 만들어 준다.
# 각각 5*5의 합성곱과 32,64개의 특징맵을 이용

conv2_flat = tf.reshape(conv2_pool, [-1,7*7*64])
full_1 = tf.nn.relu(full_layer(conv2_flat,1024))

keep_prob = tf.placeholder(tf.float32)
full1_drop = tf.nn.dropout(full_1,keep_prob=keep_prob)

y_conv = full_layer(full1_drop,10)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# fully connect layer
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for i in range(STEPS):
		batch = mnist.train.next_batch(MINBATCH_SIZE)

		if i % 100 == 0:
			train_accuracy = sess.run(accuracy, feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
			print("step {}, training accuracy {}".format(i,train_accuracy))

		sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    X = mnist.test.images.reshape(10, 1000, 784)
    Y = mnist.test.labels.reshape(10, 1000, 10)
    test_accuracy = np.mean(
        [sess.run(accuracy, feed_dict={x: X[i], y_: Y[i], keep_prob: 1.0}) for i in range(10)])

print("test accuracy: {}".format(test_accuracy))
