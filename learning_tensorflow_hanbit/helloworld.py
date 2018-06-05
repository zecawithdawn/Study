import tensorflow as tf

h = tf.constant("hello")
w = tf.constant("world")
hw = h + w
# tensor가 아닌 python으로도 같은 코드를 만들수 있지만
# print를 이용해 출력할 경우 tensor는
# Tensor("add:0",shape = (),dtype = string)
# 의 형태로 출력되는데 이에대한 설명은 다음장에

with tf.Session() as sess :
	ans = sess.run(hw)
# Session 객체는 외부의 텐서플로 연산에 대한 인터페이스 역할을 함
# 정의한 그래프를 실행시켜준다
print(ans)