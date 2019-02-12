from __future__ import print_function
#기능 나중에 알아보기

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

#data creation
'''
1. 문장을 읽으면서 unique한 알파벳들을 리스트에 저장
2. 이를 기반으로 dictionary 만들고, 이 개수를 기반으로 필요한 hyper parameter 값 설정
3. 본인이 정해준 sequence length만큼 주어진 글 자르기 (sample data 형성 과정)
4. 문장(char의 배열)을 dictionary의 index(숫자의 배열)로 변형해줌.
5. one-hot encoding으로 각 index를 0과 1을 활용하여 나타내줌.

-> objective: 이 과정이 sentence가 뭐가 오든 자동으로 이뤄지게 한다.
'''

# 1번 과정
char_set = list(set(sentence)) 

# 2번 과정
#(dictionary 형성)
char_dic = {w: i for i, w in enumerate(char_set)}

#hyper parameters
dic_size = len(char_set)  # RNN input size(one hot size)   ex. [1,0,0,0,0]
hidden_size = len(char_set)  #RNN output size. one hot형태로 바꿔줘서 데이터를 넣어주고 output도 one hot으로 받아서 알파벳 예측하려는 거니까 
                             #input과 동일한 사이즈   ex. [0.1, 0.1, 0.1, 0.1, 0.6]
num_classes = len(char_set)  #데이터를 one-hot으로 바꿔줄 때 몇개짜리 one-hot이 필요한지 알려줘야 하므로 이때 사용
sequence_length = 10  # Any arbitrary number
learning_rate = 0.1

# 3번 과정
dataX = []
dataY = []
for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1: i + sequence_length + 1]
    print(i, x_str, '->', y_str)  #우리의 objective을 보여줌

# 4번 과정 char -> index    
    x = [char_dic[c] for c in x_str]  # x str to index    ex: if you wan -> [0,1,2,3,4,5,2,6,7,8]
    y = [char_dic[c] for c in y_str]  # y str to index

    dataX.append(x)
    dataY.append(y)

batch_size = len(dataX)

X = tf.placeholder(
    tf.int32, [None, sequence_length]) 
Y = tf.placeholder(tf.int32, [None, sequence_length])

# 5번 과정 One-hot encoding
X_one_hot = tf.one_hot(X, num_classes)
print(X_one_hot)  # check out the shape   1이라는 index가 01000000...이런 식의 one hot encoding을 거치므로 1개짜리가 1*n개짜리 행렬된다.
print(X_one_hot.shape)


# Make a lstm cell with hidden_size (each unit output vector size)
cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
multi_cells = rnn.MultiRNNCell([cell] *2, state_is_tuple=True)


# RNN을 deep하게
#cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
#multi_cells = rnn.MultiRNNCell([cell]*2, state_is_tuple=True)  #2: 몇개 쌓을건지

# outputs: unfolding size x hidden size, state = hidden size
outputs, _states = tf.nn.dynamic_rnn(multi_cells, X_one_hot,initial_state=initial_state, dtype=tf.float32)

# FC layer(softmax)
# softmax 넣을 수 있게 reshape
X_for_softmax = tf.reshape(outputs, [-1, hidden_size])  #one hot encoding이니까 그거에 맞춰서(hidden_size) 개수는 알아서(이 경우는 batch size)

#softmax
outputs = tf.contrib.layers.fully_connected(X_for_softmax, num_classes, activation_fn=None)

'''
X_for_softmax=tf.reshape(outputs, [-1, hidden_size])
softmax_w= tf.get_variable("softmax_w", [hidden_size, num_classes])
softmax_b= tf.get_variable("softmax_b", [num_classes])
outputs = tf.matmul(X_fot_softmax, softmax_w)+ softmax_b
'''


# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

# All weights are 1 (equal weights)
weights = tf.ones([batch_size, sequence_length])

# loss func으로 sequence_loss 이용
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
mean_loss = tf.reduce_mean(sequence_loss)

# optimizer
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss)


#### Training
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(500): # 500 epoch
    _, l, results = sess.run(
        [train_op, mean_loss, outputs], feed_dict={X: dataX, Y: dataY})
    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        if j==169:
            print(i, j, ''.join([char_set[t] for t in index]), l)

### Let's print the last char of each result to check it works
results = sess.run(outputs, feed_dict={X: dataX})
for j, result in enumerate(results):
    index = np.argmax(result, axis=1)
    if j is 0:  # print all for the first result to make a sentence
        print(''.join([char_set[t] for t in index]), end='')
    else:
        print(char_set[index[-1]], end='')

'''
0 167 tttttttttt 3.23111
0 168 tttttttttt 3.23111
0 169 tttttttttt 3.23111
…
499 167  of the se 0.229616
499 168 tf the sea 0.229616
499 169   the sea. 0.229616
g you want to build a ship, don't drum up people together to collect wood and don't assign them tasks and work, but rather teach them to long for the endless immensity of the sea.
'''