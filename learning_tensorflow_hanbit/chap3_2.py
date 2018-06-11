import tensorflow as tf
import numpy as np
'''
# variable은 모델의 매개변수를 조정하는 역할을 한다.
# 세션이 실행될때마다 '리필'되는 다른 텐서객체랑은 달리 고정된 상태 유지

init_val= tf.random_normal((1,5),0,1)
# tf.random_normal(shape,mean,stddev)
# 정규분포를 따르는 난수 생성 (mean = 평균,stddeb = 표준편차)

var = tf.Variable(init_val,name='var')
init = tf.global_variables_initializer()
# variable 사용시 변수를 '할당' 하고 '초기화' 하는 과정이 꼭 필요
'''

'''
# placeholder는 입력값을 공급하기 위한 구조
# 나중에 데이터로 채워질 '빈' 변수라고 생각하면된다.
# 그래프가 실행되는 시점에 입력 데이터를 밀어 넣음

ph = tf.placeholder(tf.float32,shape=(None,10))
# shape를 선택적으로 사용 할 수 있는데 값이 지정되지 않거나 None을 쓰는경우
# '모든' 크기의 데이터를 받을 수 있다. 
# 행렬에서 샘플데이터의 개수를 의미하는 행은 'None'으로 특징의 길이로 쓰이는
# 열은 '고정'된 값으로 주로 사용 

sess.run(s,feed_dict={x: X_data, w: w_data})
# session이 실행될때 feed_dict 함수를 이용 값을 밀어 넣는다
# feed_dict의 딕셔너리 키는 placeholder 변수명에 대응
'''

'''
x_data = np.random(2000,3)
w_real = [0.3,0.5,0.1]
b_real = -0.2
noise = np.random.randn(1,2000)*0.1
y_data = np.matmul(w_real,x_data.T)+ b_real + noise
# numpy를 사용 예제에 필요한 합성 데이터를 생성 3개의 특징을 가진 샘플 2000개
# 가중치 0.3,0.5,0.1을 넣고 편향 -0.2와 가우시안 노이즈를 더한것

NUM_STEPS = 10
# 반복 실행 횟수

g = tf.Graph()
# import tensorflow as tf 라인의 의해 default한 graph가 이미 생성되어있는데
# 코드 내에서 여러개의 그래프를 관리 할때 사용

wb_ = []

with g.as_default():
	# 새로 생성한 그래프를 이 with 문 내에서 사용하기 위해 default를 변경시켜준다.

	x = tf.placeholder(tf.float32,shape=[None,3])
	y_true = tf.placeholder(tf.float32,shape=None)
	# 추후에 값을 밀어 넣어줄 placeholder들임 가중치(w)와 편향(b)을 구할 변수들이기 때문에
	# x는 [None,3]의 형태로 선언되었음

	# 이름스코프를 이용해 추론,손실함수정의,학습객체 설정을 각각 묶었다

	with tf.name_scope('inference') as scope:
		w = tf.Variable([[0,0,0]],dtype = tf.float32,name='weights')
		b = tf.Variable(0,dytpe = tf.float32,name='bias')
		y_pred = tf.matmul(w,tf.transpose(x)) + b
		# y = wx + b

	with tf.name_scope('loss') as scope:
		loss = tf.reduce_mean(tf.square(y_true,y_pred))
		# 손실함수는 실제값과 예상값의 차를 제곱한 형태 (평균제곱오차(MSE))를 사용

	with tf.name_scope('train') as scope:
		learning_rate = 0.5
		# 학습률은 각각의 갱신 반복이 얼마나 적극적으로 이루워 지는가(연산을 통해 정해진 step의 얼마만큼을 이동할 것인가)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		# GradientDescentOptimizer() 함수를 이용 최적화 함수를 생성한다.
		train = optimizer.minimize(loss)
		# optimizer.minize()함수를 이용 변수를 갱신하는 학습연산을 생성한다.
		
	init = tf.global_variables_initializer()
	# 세션을 실행하기전 할당해놓은 변수들을 초기화

	with tf.Session as sess:
		sess.run(init)
		for step in range(NUM_STEPS):
			sess.run(train,{x: x_data,y_true: y_data})
			if (step % 5 == 0):
				print(step,sess.run([w,b]))
				wb_.append(sess.run([w,b]))
		print(10, sess.run([w,b]))

'''











