#!/usr/bin/env python
# coding: utf-8

# In[32]:


#1. 라이브러리 import
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#2. 데이터 셋 섞음(train/validation/test data set)
#함수 정의해서 이용00
def shuffle_data(x_train, y_train):
    temp_index=np.arange(len(x_train))
        #arange 기능: np.arange(10)이면 [0 1 2 3 4 5 6 7 8 9] 수열 생성
        #결국 여기서는 0~(len(x_train)-1)의 값으로 이뤄진 수열 생성
    
    #random shuffle index
    np.random.shuffle(temp_index)

    #re_arragne x and y data with random shuffle index
    x_temp=np.zeros(x_train.shape)
    y_temp=np.zeros(y_train.shape)
    x_temp=x_train[temp_index] #섞어준 순서대로 다시 넣기
    y_temp=y_train[temp_index] #섞어준 index기준으로 재입력하기 때문에 값이 섞여진 효과

    return x_temp, y_temp

#3. Train set 생성하기
def main():
    num_points = 5000 #(X,Y)데이터 개수
    vectors_set=[]
    for i in range(num_points):
        x1=np.random.normal(.0, 1.0)  #평균 0, 표준편차 1인 정규분포에서 임의로 값 추출
        y1=np.sin(x1)+np.random.normal(0., 0.1)  #sin(x1)에 평균 0, 표준편차 0.1인 정규분포에서 임의로 값 뽑아서 오차 더해주기
        vectors_set.append([x1,y1])
    return vectors_set

v1=main()
x_data=[v[0] for v in v1]
y_data=[v[1] for v in v1]
    
#green 색(g)에 둥근 점(o)으로 시각화
plt.plot(x_data, y_data, 'go')
plt.legend()
plt.show()
    
#배치 수행단위
BATCH_SIZE=100
BATCH_NUM=int(len(x_data)/BATCH_SIZE)
    
#데이터를 세로로(한개씩) 나열한 형태로 reshape
x_data=np.reshape(x_data, [len(x_data),1])
y_data=np.reshape(x_data, [len(y_data),1])
    
#총개수는 정해지지 않았고 1개씩 들어가는 placeholder 생성
input_data=tf.placeholder(tf.float32, shape=[None,1])
output_data=tf.placeholder(tf.float32, shape=[None,1])

#레이어 간 weight 정의 후 랜덤값으로 초기화
W1=tf.Variable(tf.random_uniform([1,5],-1.0, 1.0))
W2=tf.Variable(tf.random_uniform([5,3],-1.0, 1.0))
W_out=tf.Variable(tf.random_uniform([3,1], -1.0, 1.0))

#레이어의 노드가 하는 계산, 이전 노드와 현재 노드의 곱셈
#비선형함수로 sigmoid 추가
hidden1= tf.nn.sigmoid(tf.matmul(input_data, W1))  #기본적인 선형 모델(행렬곱)에 활성화 함수(sigmoid)인 히든레이어 추가해준다
hidden2= tf.nn.sigmoid(tf.matmul(hidden1, W2))
output= tf.matmul(hidden2, W_out)

#cost func
cost=tf.reduce_mean(tf.square(output-output_data))  #output: 신경망에서 출력된 값(예측),   output_data: 실제 데이터(정답)

#optimizer-최적화 함수
optimizer= tf.train.AdamOptimizer(0.01)   # 0.01= learning rate
    #예전에는 train= tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)로 했던거랑 달리
    #여기선 adamoptimizer이용

#train
train= optimizer.minimize(cost)

#변수 사용 준비
init= tf.global_variables_initializer()

#세션 열고 init 실행
sess=tf.Session()
sess.run(init)

#반복하면서 값 업데이트
for step in range(5001):
    index=0
    
x_data, y_data= shuffle_data(x_data, y_data)

# 배치크기만큼 학습 진행
for batch_iter in range(BATCH_NUM -1):
    #origin
    feed_dict= {input_data: x_data[index: index+BATCH_SIZE],
                output_data: y_data[index: index+BATCH_SIZE]}
    sess.run(train, feed_dict=feed_dict)
    index += BATCH_SIZE

#화면에 학습 진행상태 출력
#최초 100회까지는 10마다, 이후는 100회에 한번씩
if(step%100==0 or (step<100 and step%10==0)):
    print("Step=%5d, Loss Vlaue=%f" %(step, sess.run(loss,feed_dict=feed_dict)))
        


# In[ ]:


'''
append 함수 기능: 주어진 리스트 맨 마지막에 해당 데이터 추가
fruits = ["apple", "banana", "cherry"]
fruits.append("orange")
print(fruits)
출력 결과: ['apple', 'banana', 'cherry', 'orange']
'''


# In[ ]:




