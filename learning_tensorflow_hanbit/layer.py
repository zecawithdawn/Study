# CNN 모델에 대한 설명은
# https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/ 여기 참고

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)
# 완전연결계층(fully connect) 나 합성곱계층(convolution)의 가중치 지정
# stddev는 표준 편차로 작을 수록 평균값이랑 비슷
# Truncated 무작위로 초기화 시킴
# 상수로 초기화 하는 대신 무작위값으로 초기화하면 다양하고 풍부한 표현을 학습할수 있음
# 무작위 초기화값에 stddev 처럼 범위를 지정하면 효율적으로 수렴할수 있음

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)
# 완전연결계층 이나 합성곱계층의 편향값(bias) 정의 둘다 0.1로 초기화

def conv2d(x, W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
# 합성곱 정의, 건너뜀 없는 완전한 합성곱으로 입력과 같은 출력
# x 는 특징맵 -> 각 계층에서 필터로 처리되어 나온 결과맵
# 			  x 의 모양은 [none, 28, 28, 1] -> 이미지개수는 모름, 28*28픽셀, 색 채널은 1개
# W 는 필터 -> 작은 슬라이딩 윈도들의 가중치의 '집합'
# 			  가중치의 모양은 [5,5,1,32] -> 5*5는 영역(사용될 윈도의 크기),32는 특징맵의수 합성곱 필터의 수
# strides는 x위에서 필터W의 공간적 이동을 제어함 [1,1,1,1]의 의미는 필터가 각차원에서 한픽셀 간격을로 입력 데이터에 적용
# 완전한 합성곱에 대응한다.(입력 크기 그대로 전부 적용한다.) strides를 더 작게 할수도 있지만 보통 이렇게 사용
# padding SAME -> X의 크기를 연산후에도 똑같은 크기로 유지하겠다.

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
# 최댓값 풀링을 통해 데이터의 크기를 줄여준다.
# '계산된 특징'의 이미지내에서 이동으로 영향을 받지 않기 위해
# ksize 인수는 풀링의 크기를 정의
# strides인수는 x에서 움직이는 풀링 조각이 얼마나 크게 건너뛸지 혹은 미끄러질지를 제어 
# 2*2이기 때문에 높이와 폭은 1/2이 됨(2칸 건너뛰기 하므로)

def conv_layer(input, shape):
	W = weight_variable(shape)
	b = bias_variable([shape[3]])
	return tf.nn.relu(conv2d(input, W)+b)
# 실제 합성곱계층으로 정의된 conv2d에 bias를 더하고 relu로 활성화 함수를 걸어준다.

def full_layer(input,size):
	in_size = int(input.get_shape()[1])
	W = weight_variable([in_size,size])
	b = bias_variable([size])
	return tf.matmul(input,W)+ b
# 편향을 적용한 완전 연결 계층
# 활성화 함수가 없음 -> 처음과 끝의 크기가 같아서

