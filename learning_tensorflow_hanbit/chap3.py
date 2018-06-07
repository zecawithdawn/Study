import tensorflow as tf
# 텐서플로우는 연산 그래프 형태이다
# 위의 구문이 쓰인 시점부터 비어 있는 기본 그래프가 생성된것
# 노드와 꼭지점의 수가 0인 그래프를 말하는듯

a = tf.constant(5)
b = tf.constant(2)
c = tf.constant(3)
# a,b,c 라는 상수값을 출력한다.

d = tf.multiply(a,b)
e = tf.add(c,b)
f = tf.subtract(d,e)
# tf.<operator> '+,-,*,/,**, ...' 등은 연산자를 대신해서 쓸수도 있다.


with tf.Session as sess:
# sess = tf.Session()
# session은 객체와 메모리가 할당되어 있는 실행 환경 사이를 연결해주고 중간 결과를
# 저장하고 최종결과를 작업 환경으로 보내준다.
# 메서드가 호출되면 출력이 나와야 하는 노드에서 시작해(위에서는 f) 역방향으로 처리
# 하며 의존관계 집합에 따라 실행되어야 하는 노드의 연산을 수행한다.

	fetchs = [a,b,c,d,e,f]
	outs = sess.run(fetchs)
# with 절로 위에서와 같이 리스트로 세션으로 넘길수 있고 입력리스트 내 순서대로 세션이 실행

# sess.close()
# 연산작업이 마무리되면 세션을 닫아주는 것이 좋다.
# with 구문은 종료뒤 자동으로 close 된다.

print("outs = {}".format(outs))


