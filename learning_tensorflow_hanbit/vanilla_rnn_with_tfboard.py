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

# 텐서보드 모델 요약을 저장할 위치
LOG_DIR = "logs/RNN_with_summaries"

# 입력과 레이블을 위한 플레이스홀더 생성
_inputs = tf.placeholder(tf.float32,shape=[None, time_steps, element_size],name='inputs')
y = tf.placeholder(tf.float32, shape=[None, num_classes],name='labels')

batch_x, batch_y = mnist.train.next_batch(batch_size)
# 28개의 시퀀스를 얻기 위해 각 데이터를 28픽셀 형태로 변환
batch_x = batch_x.reshape((batch_size, time_steps,element_size))

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

def rnn_step(previous_hidden_state,x):
	current_hidden_state = tf.tahn(tf.matmul(previous_hidden_state, Wh) + tf.matmul(x, Wx) + b_rnn)
	return current_hidden_state

processed_input = tf.transpose(_input, perm=[1,0,2])
initial_hidden = tf.zeros([batch_size,hidden_layer_size])
all_hidden_states = tf.scan(rnn_step,processed_input,initializer=initial_hidden,name='states')

with tf.name_scope('linear_layer_weights') as scope:
	with tf.name_scope("W_linear"):
		Wl = tf.Variable(tf.truncated_normal([hidden_layer_size,num_classes],mean=0,stddev=0.01))
		variable_summaries(Wl)
	with tf.name_scope("Bias_linear"):
		bl = tf.Variable(bl)

# 상태 벡터에 선형 계층 적용
def get_linear_layer(hidden_state):
	return tf.matmul(hidden_state, Wl) + bl

with tf.name_scope('linear_layer_weights') as scope:
	all_outputs = tf.map_fn(get_linear_layer, all_hidden_states)
	# 최종 결과
	output = all_outputs[-1]
	tf.summary.histogram('outputs',output)

with tf.name_scope('cross_entropy'):
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entopy_with_logits(logits=output, labels =y))
	tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
	train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

