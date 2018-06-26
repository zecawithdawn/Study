import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 데이터를 내려 받아서 사용하는 것이 아니라 내장 유틸리티에서 바로 사용

DATA_DIR = 'tmp/data'
NUM_STEPS = 1000
MINIBATCH_SIZE = 100
# 학습에 사용될 여러 상수 정의

data = input_data.read_data_sets(DATA_DIR,one_hot=True)
# read_data_sets() 메서드는 데이터를 내려받아 로컬에 저장하여 사용할 수 있도록 해준다.

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))
# Variable은 연산과정 중에 조작되는 값
# placeholder는 연사 그래프가 실행 때 제공되어야 하는 값
# x는 이미지 자체이고 연산될때 제공됨
# 784는 28 * 28 px의 이미이고 None은 얼마나 많은 이미지를 넣을지는 지정해 놓지 않겠다는 의미

y_true = tf.placeholder(tf.float32, [None,10])
y_pred = tf.matmul(x, W)
# 지도학습(supervised learning)에서 쓰이는 모델로 예측 레이블과 정답 레이블을 지정
# tf.matmul matrix multiply

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
# 교차 엔트로피라는 유사성 척도를 사용, 모델의 출력 값이 각 분류에 대응되는 확률일 때 일반으로 사용
# loss function(손실함수)라고 부르기도 한다

gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 학습 방법 손실, 손실 함수의 값을 최소화하는 방법에 관한 것이다. 일반적으로 GradientDescent(경사하강법)을 사용
# 0.5는 학습률로 경사 하강 최적화 함수가 전체 손실이 감소되록 가중치를 이동 시킬때 얼마나 빨리 이동할지를 제어

correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))
# 테스트를 위해 평가 과정을 정의해야 함
# 여기서는 정확하게 분류된 테스트 데이터의 비율을 사용한다.


with tf.Session() as sess :
	# 학습
	sess.run(tf.global_variables_initializer())
	# 모든 변수를 초기화 한다.
	# 초기화 과정은 머신러닝과 최적화에 특정한 영향을 준다.

	for _ in range(NUM_STEPS):
		batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
		sess.run(gd_step, feed_dict={x: batch_xs, y_true:batch_ys})
	# 경사하강법에서의 모델을 '올바른 방향'으로 여러번 이동 하는 단계(NUMSTEPS)로 구성된다.
	# 각 단계에서 데이터 관리 모듈에 레이블을 포함한 데이터(예제)를 뭉치(MINBATCH_SIZE)를 요청하고 이를 학습 모듈에 넘겨준다.

	# 테스트
	ans = sess.run(accuracy, feed_dict={x:data.test.images,y_true: data.test.labels})
	# feed_dict는 placeholder가 포함된 연산을 수행하고자 할 때(Step) 마다 placeholder에 값을 밀어 넣는것이다.
	
print("Accuracy: {:.4}%".format(ans*100))