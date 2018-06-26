import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot =True)

# 매개변수 정리
element_size = 28
time_steps = 28 
num_classes = 10
batch_size = 128
hidden_layer_size = 128
# element_size - 시퀀스 벡터 각각의 차원, 열또는 행의 픽셀크기인 28
# time_steps - 한 시퀀스 내에 들어있는 원소의 수
# CNN에서 mnist데이터는 28*28 의 784 px의 벡터로 저장 되었는데
# 여기서는 행열과 열중 하나를 선택해 하나는 데이터 하나는 시간축으로 표현했다.

LOG_DIR = "logs/RNN_with_summaries"
# 텐서보드 모델 요약을 저장할 위치

_inputs = tf.placeholder(tf.float32,shape=[None, time_steps, element_size],name='inputs')
y = tf.placeholder(tf.float32, shape=[None, num_classes],name='labels')
# 입력과 레이블을 위한 플레이스홀더 생성
# 플레이스 홀더는 세션이 시작될때 feed_dict로 값이 주워진다.

def variable_summaries(var):
	with.tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean )))
		tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_min(var))
		tf.summary.histogram('histogram', var)
# Log에 적어줄 요약을 위한 몇몇 연산
# mean 평균 stddev 표준 편차 최댓값 최솟값
# histogram은 TensorBoard에 데이터 분포를 시각화 하기 위해 추가

with tf.name_scope('rnn_weights'):
	with tf.name("W_x"):
		Wx = tf.Variable(tf.zeros([element_size, hidden_layer_size]))
		variable_summaries(Wx)
	with tf.name_scope("W_h"):
		Wh = tf.Variable(tf.zeros([hidden_layer_size,hidden_layer_size]))
		variable_summaries(Wh)
	with tf.name_scope("Bias"):
		b_rnn = tf.Variable(tf.zeros([hidden_layer_size]))
		variable_summaries(b_rnn)
# RNN모델에 들어갈 변수들 (플레이스 홀더 아님 나중에 session이 실행되면 변경되는 값)
# 을 초기화 시켜준다  

def rnn_step(previous_hidden_state,x):
	current_hidden_state = tf.tahn(tf.matmul(previous_hidden_state, Wh) + tf.matmul(x, Wx) + b_rnn)
	return current_hidden_state
# current = tahn(previous * Wh + x * Wx + Bias)
# 로 RNN 모델을 정의

# scan 함수로 입력값 처리
# 입력의 형태 (batch_size, time_steps, element_size)
processed_input = tf.transpose(_inputs, perm=[1,0,2])
# 변형된 입력의 형태 (timestep,batch_size,element_size)
# perm은 인수를 변경할 축을 지정함
initial_hidden = tf.zeros([batch_size,hidden_layer_size])
# 시간의 흐름에 따라 상태 벡터 구하기
all_hidden_states = tf.scan(rnn_step,processed_input,initializer=initial_hidden,name='states')
# tf.scan은 같은 연산을 명시적으로 복제해가면서 루프를 풀어놓지 않고 연산 그래프에
# 루프를 사용하게 해준다.
# 시간의 흐름에따른 중간 누적값을 모두 반환함

# 출력에 저용할 가중치
# RNN은 각 시간 단계에 대한 상태 벡터에 가중치를 곱하여 데이터의 새로운 표현인 출력벡터를 얻는다.
 with tf.name_scope('linear_layer_weights') as scope:
	with tf.name_scope("W_linear"):
		Wl = tf.Variable(tf.truncated_normal([hidden_layer_size,num_classes],mean=0,stddev=0.01))
		variable_summaries(Wl)
	with tf.name_scope("Bias_linear"):
		bl = tf.Variable(tf.truncated_normal([num_classes],mean =0,stddev =0.01))
		variable_summaries(bl)
# 상태 벡터에 선형 계층 적용
def get_linear_layer(hidden_state):
	return tf.matmul(hidden_state, Wl) + bl
with tf.name_scope('linear_layer_weights') as scope:
	# 시간에 따라 반복하면서 모든 RNN 결과에 선형 계층 적용
	all_outputs = tf.map_fn(get_linear_layer, all_hidden_states)
	# map_fn은 호출된 요소를 처름부터 마지막까지 끝까지 적용함
	# 팩토리 함수를 만든 효과
	output = all_outputs[-1]
	# 최종 결과
	# RNN은 앞의 state의 영향을 받아 현재 값을 계산하기 때문에 가장 마지막 값이
	# 결과 값임 그래서 -1 표현을 사용
	tf.summary.histogram('outputs',output)

# RNN분류
with tf.name_scope('cross_entropy'):
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entopy_with_logits(logits=output, labels =y))
	tf.summary.scalar('cross_entropy', cross_entropy)
# 손실 함수 교차 엔트로피 방법 사용했음
with tf.name_scope('train'):
	train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)
# 최적화 
# RMSPropOptimizer 강력한 경사 하강법 알고리즘
### 이건 추후에 공부
with tf.name_scope('accuracy'):
	correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(output,1))
	accuracy = (tf.reduce_mean(tf.cast(correct_prediction,tf.float32))) * 100
	tf.summary.scalar('accuracy',accuracy)
# 예측
merged = tf.summary.merge_all()
# 텐서보드에 사용할 요약 추가하기 위해 위의 내용을 합친다
# 내용이 병합 되어있는 '텐서'를 얻을 수 있다.
### merged(cross_entropy,train_step,crrect_prediction,accuracy)같은 함수 형태로 봐도 되는건가? 

# 테스트를 위한 데이터 생성
test_data = mnist.test.image[:batch_size].reshape((-1, time_steps,element_size))
test_label = mnist.test.labels[:batch_size]
# reshape는 numpy패키지의 함수로 -1 을 쓸수있는데
# -1은 다른 요소에서 남은 값을 사용한다
# ex 12 0 0 의 배열이  (2 2 -1)로 reshape 해준다면 실제값은 (2,2,3)이다

# 위에서 생성한 모델을 실행
with tf.Session() as sess:
	train_writer = tf.summary.FileWriter(LOG_DIR + '/train',gragh=tf.get_default_graph())
	test_writer = tf.summary.FileWriter(LOG_DIR + '/test',gragh = tf.get_default_graph())
	# LOG_DIR에 텐서보드에서 사용할 요약을 기록
	
	sess.run(tf.global_variables_initializer())
	# session을 시작하기 이전에 항상 초기화를 먼저
	for i in range(10000):
		# 10000번 학습을 실행 할것임
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		# next_batch는 train 데이터 셋에서 batch_size 만큼의 랜덤 데이터를 가져와서 넣겠다는 뜻
		batch_x = batch_x.reshape((batch_size, time_steps,element_size))
		# 데이터 값은 직렬 형태의 784,0 의 형태이므로
		# 28개의 시퀀스를 얻기 위해 각 데이터를 28픽셀 형태로 변환
		# batch를 128 * 28 * 28 형태의 배열로 변경해줌
		summary,_ = sess.run([merged, train_step], feed_dict = {_inputs:batch_x,y:batch_y})
		# [merged와 train_step]의 값을 얻기 위해 데이터를 밀어 넣음
		### 백 프로퍼게이션은 위의 최적화 함수에 포함되있는건가??
		train_writer.add_summary(summary,i)
		if i % 1000 == 0 :
			acc,loss = sess.run([accuracy,cross_entropy],feed_dict={_inputs:batch_x,y:batch_y})
			print("Iter "+ str(i) + ", Minibatch Loss = " + "{:.6f}".format(loss) + ",Traing Accuracy=" + "{:.5f}".format(acc))

		if i % 100 = 0 :
		summary, acc = sess.run([merged, accuracy], feed_dict={_inputs: test_data,y = test_label})
		test_writer.add_summary(summary,i)

	test_acc = sess.run(accuracy, feed_dict={_inputs: test_data,y:test_label})
	print("Test Accuracy:",test_acc)
### for 문이 끝나면 학습이 끝나는 것이니 with문 내에서 input을 만들어서
### 내가 직접 값을 넣고 컴퓨터가 무슨 글자인지 알려주는 코드를 추가해보자
