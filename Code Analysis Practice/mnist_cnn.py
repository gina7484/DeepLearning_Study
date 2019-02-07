#출처: hunkim/DeepLearningZeroToAll/lab-11-1-mnist_cnn.py
#URL: https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-11-1-mnist_cnn.py
#Youtube URL: https://www.youtube.com/watch?v=pQ9Y9ZagZBk&feature=youtu.be

#직접 돌리기보다는 어려운 CNN만들기 전에 좀 더 단순한 예제 코드보면서 이해하기 위한 용도로 이용

# Lab 11 MNIST and Convolutional Neural Network

#전체적인 구조
# conv - relu - pooling - conv - relu - pooling - fc

import tensorflow as tf
import random
# import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28x28x1 (black/white라서 마지막에 1)
Y = tf.placeholder(tf.float32, [None, 10])


########## 첫번째 conv layer ##########

#1. 사용하려는 필터 크기랑 개수 정해주고 초기화
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))  
                                 #[3,3,1,32]에서 3*3*1은 필터크기, 32는 필터개수

#2. conv layer 어떻게 통과시킬건지 정해주기    
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
#위의 의미:
    #input img은 X_img이다. (28*28*1 사이즈)
    #W1 필터를 쓸거임.
    #stride=[1,1,1,1] 두번째랑 세번째 parameter에 의해 수평,수직으로 각각 한칸씩 이동
    #padding='SAME'라서 입력의 이미지 사이즈 28*28과 같아지도록!
    #ceil(float(in_height) / float(strides))
    #그래서 conv layer 통과시키면    Conv     -> (?, 28, 28, 32)

#3. conv layer 출력값에 relu 씌워주기
L1 = tf.nn.relu(L1)

#4. subsampling(pooling-여기서는 pooling 방법 중 max pooling 이용)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
    #ksize=[1,2,2,1] //kernel size=2*2 그래서 2*2의 4칸 중에서 제일 큰 값을 뽑는다
    #stride=[1,2,2,1] //2칸씩 수평, 수직으로 이동한다
    #padding='SAME' -> 이에 대한 자세한 계산 방법은 https://www.tensorflow.org/api_guides/python/nn#Convolution 

'''
Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)  //conv layer 통과시켰을 때 결과
Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)    //relu 씌워줬을 때 결과 
Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32) //pooling 이후 결과이자 두번째 conv layer 입력값
마지막 결과인 14*14*32를 두번째 conv layer 입력값으로 이용한다
'''

########## 두 번째 conv layer ##########

#1. 사용하려는 필터 크기랑 개수 정해주고 초기화
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
                                 #[3,3,32,64]에서 3*3*32은 필터크기, 64는 필터개수

#2. conv layer 어떻게 통과시킬건지 정해주기
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    #    Conv     -> (?, 14, 14, 64)
    # padding='SAME'하면 out_height = ceil(float(in_height) / float(strides))니까 14/1해서 14가 된다 -width도 마찬가지

#3. conv layer 출력값에 relu 씌워주기
L2 = tf.nn.relu(L2)
    #    relu     -> (?, 14, 14, 64)

#4. subsampling(pooling-여기서는 pooling 방법 중 max pooling 이용)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
    #    pool     -> (?, 7, 7, 64)
    
#5. fc에 넣기 위한 reshape 과정
L2_flat = tf.reshape(L2, [-1, 7 * 7 * 64])
    # -1이 n개의 값이라고 하는데...?

'''
Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
Tensor("Reshape_1:0", shape=(?, 3136), dtype=float32)
'''

########## Final FC 7x7x64 inputs -> 10 outputs ##########
W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L2_flat, W3) + b

########## define cost/loss ##########
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
    # softmax 함수에 대한 자세한 설명: https://bcho.tistory.com/1154
    # softmax:  classification 알고리즘중의 하나로, 들어온 값이 어떤 분류인지 구분해주는 알고리즘이다. 
                #예를 들어 A,B,C 3개의 결과로 분류해주는 소프트맥스의 경우 결과값은 [0.7,0.2,0.1] 와 같이 각각 A,B,C일 확률을 리턴해준다.
    # softmax_cross_entropy_with_logits: 
        # 소프트맥스 함수에 대한 코스트 함수는 크로스엔트로피 (Cross entropy) 함수의 평균을 이용

########## define optimizer ##########
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

########## initialize ##########
sess = tf.Session()
sess.run(tf.global_variables_initializer())

########## train my model ##########
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += (c / total_batch)
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

########## Test model and check accuracy ##########
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels}))

########## Get one and predict ##########
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r + 1]}))