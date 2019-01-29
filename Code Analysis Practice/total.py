#!/usr/bin/env python
# coding: utf-8

# In[11]:


#1.필요한 라이브러리 가져오기
import numpy as np
import pandas
import tensorflow as tf

#2. load CSV file as matrix
train_csv_data = pandas.read_csv('train.csv').values
test_csv_data=pandas.read_csv('test.csv').values
test_csv_sub=pandas.read_csv('gender_submission.csv').values

#3. 데이터 전처리 과정
#(3-1)남녀가 male, female로 되어있는데 이거 바꿔주기(데이터 전처리 과정1)
#train set
for i in range(len(train_csv_data)):
    #print(train_csv_data[i,4])
    if train_csv_data[i,4]=='male':
        train_csv_data[i,4]=1
    else:
        train_csv_data[i,4]=0
        
#test set
for i in range(len(test_csv_data)):
    #print(test_csv_data[i,3])
    if test_csv_data[i,3]=='male':
        test_csv_data[i,3]=1
    else:
        test_csv_data[i,3]=0
        

#(3-2)승선항에 대한 정보(영문) -> 수치화해주기 (데이터 전처리 과정2)
'''
바꿔주려는 방법
empty ->0
S ->1
C ->2
Q ->3
'''

#training set -각 행에서 11번째 열이 승선항에 대한 정보
for i in range(len(train_csv_data)):
    if train_csv_data[i,11]=='S':
        train_csv_data[i,11]=1
    elif train_csv_data[i,11]=='C':
        train_csv_data[i,11]=2
    elif train_csv_data[i,11]=='Q':
        train_csv_data[i,11]=3
    if np.isnan(train_csv_data[i,11]):
        train_csv_data[i,11]=0
        
#test set -각 행에서 10번째 열이 승선항에 대한 정보
for i in range(len(test_csv_data)):
    if test_csv_data[i,10]=='S':
        test_csv_data[i,10]=1
    elif test_csv_data[i,10]=='C':
        test_csv_data[i,10]=2
    elif test_csv_data[i,10]=='Q':
        test_csv_data[i,10]=3
    if np.isnan(test_csv_data[i,10]):
        test_csv_data[i,10]=0

        
        
#4. 데이터셋 파일에서 필요한 항목만 실제 학습용 데이터로 추려내기 -> 나중에 함수 다 짜준 다음에 얘를 feed_dict로 입력해줄 거임

#Pclass, sex, sibsp, parch, embarked 이 5가지 데이터를 입력 X로 해서 생존여부 Y를 예측하려고 함.


#Train set
X_PassengerData = train_csv_data[:,[2, #Pclass
                                    4, #Sex
                                    6, #Sibsp
                                    7, #Parch
                                    11 #Embarked
                                    ]]
Y_Survived =train_csv_data[:,1:2] # 첫번째 :는 전체 의미함.  1:2=1이상 2미만=1

#Test set
Test_X_PassengerData = test_csv_data[:,[1, #Pclass
                                        3, #Sex
                                        5, #Sibsp
                                        6, #Parch
                                        10 #Embarked
                                        ]]
Test_Y_Survived =test_csv_sub[:,1:2]



#5. 가설식 생성하기
'''
Pclass, sex, sibsp, parch, embarked 이 5가지 데이터를 입력 X로 해서 생존여부 Y를 출력하는 가설을 생성하려고 함.
Y=W1*Xp + W2*Xs + W3*Xsib + W4*Xpar + W5*Xe + b
Y=WX+b 의 행렬형태로 나타내줄 수 있음
W: [W1 W2 W3 W4 W5]
X: [Xp
    Xs
    Xsib
    Xpar
    Xe]
그래서 여기선 행렬곱 matmul()사용
'''

#placeholder -feed_dict로 값을 외부에서 입력받기 때문에 3가지 변수형 중에 이거 써야 함
X=tf.placeholder(tf.float32, shape=[None,5])
Y=tf.placeholder(tf.float32, shape=[None,1])

# W,b 초기화 해주기
W=tf.Variable(tf.random_normal([5,1]),name='weight')
b=tf.Variable(tf.random_normal([1]), name='bias')

#hypothesis
hypothesis= tf.sigmoid(tf.matmul(X,W)+b)  #sigmoid를 씀으로써 값이 0~1사이의 실수가 된다. 후에 0.5를 기준으로 생존, 사망 결정(8번 단계)



#6. const func 만들기
cost= -tf.reduce_mean(Y* tf.log(hypothesis) + (1-Y)* tf.log(1-hypothesis))
#reduce_mean에서 두번째 parameter가 생략되었으므로 이는 단순 평균을 구해준다!
#Y는 0 or 1이므로 로그함수 형태에 의해 정답이면 cost 0에 가깝고, 틀리면 cost가 무한대에 가까운 값이 되게



#7. optimizer
train= tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)



#8. accuracy 정의(정답과 가설의 출력값 차이 비교)
predicted= tf.cast(hypothesis>0.5, dtype=tf.float32)  #hypothesis>0.5 참이면 1(생존), 아니면 사망(0)
#우리는 이 predicted(가설의 출력값)과 실제 Y(정답)이 같은 비율을 구하려고 하는 것(같은 횟수/전체 횟수)
accuracy= tf.reduce_mean( tf.cast( tf.equal(predicted,Y),dtype=tf.float32 ) )
#tf.equal(predicte,Y)가 참이면(가설이 정답을 맞췄으면) 1 반환하게 하는 구조.
#reduce_mean에서 두번째 변수가 생략되어서 단순 평균을 구해준다.


#9. Launch Graph
with tf.Session() as sess:  #linear regression 예제와 달리 sess쓰려면 이 문구 밑에다가 써줘야 함
    sess.run(tf.global_variables_initializer())
    
    #10. 반복해서 최적값 찾기
    for step in range(10000):
        cost_val, _ = sess.run([cost,train], feed_dict={X:X_PassengerData, Y: Y_Survived})
    
        if step%500==0:
            print("Step= ", step, ", Cost: ", cost_val)
        
        #11. 오버피팅의 문제
         #원래는 validation Data Set으로 했을 때 오차가 증가할 때 학습을 그만 둔다고 배웠다.
         #그래서 데이터가 많으면 주로 training data set의 일부를 validation data set으로 이용한다.
         #하지만 이 예제는 데이터 부족함.  -> validation data set 안 만들고 오버피팅 피하는 방법이 필요함
         #이 경우는 예측모델이 간단해서 cost 진척이 없으면 조기종료
        if step==0:
            previous_cost=cost_val
        else:
            if previous_cost ==cost_val:
                print("found best hypothesis when step: ", step , "\n")
                break
            else:
                previous_cost = cost_val
                
    #12. 가설검증(설명력)
    h,c,a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:X_PassengerData, Y: Y_Survived})
    print("\n Accuracy: ", a)
    
    #13. Test set으로 학습한 모델의 최종 성능 확인
    print("\n Test CSV runningResult")
    h2,c2,a2=sess.run([hypothesis, predicted, accuracy],feed_dict={X: Test_X_PassengerData, Y: Test_Y_Survived})
    print("\n Accuracy: ", a2)


# In[ ]:




