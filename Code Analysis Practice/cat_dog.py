# 코드 출처: https://www.kaggle.com/sentdex/full-classification-example-with-convnet

#라이브러리 가져오기
import numpy as np
import os           # dealing with directories
from random import shuffle
from tqdm import tqdm
import cv2      # working with, mainly resizing, images

#데이터 다운받은 다음에 주소 복사해서 디렉토리 정해주기
##주의: 윈도우에서 주소 복사하면 /이 아니라 \이라서 다 /로 바꿔줘야 함

TRAIN_DIR='C:/Users/SuperNoteJ/Desktop/deep0121/data/dogs-vs-cats-redux-kernels-edition/train/train' 
TEST_DIR = 'C:/Users/SuperNoteJ/Desktop/deep0121/data/dogs-vs-cats-redux-kernels-edition/test/test'

#img size 정해주기-사진들이 다 똑같은 사이즈가 아니라서 50*50으로 바꿔줄거임
IMG_SIZE = 50

#learning rate =0.001
LR = 1e-3  

#모델 이름 정하기
MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic') # just so we remember which saved model is which, sizes must match


#라벨링해주기
#Our images are labeled like "cat.1" or "dog.3"
#So we need to convert the images and labels to array information that we can pass through our network

#입력값: img path
def label_img(img):
    word_label = img.split('.')[-3]
    #위에서 -3은 split한 덩어리 중에서 필요한 것의 word label
    #예를 들면 cat.11.jpg로 저장되어 있으니까 .으로 나눈다음에 -1 label은 jpg, -2 label은 11, -3 label은 cat
    # conversion to one-hot array [cat,dog]
    if word_label == 'cat': return [1,0]
    elif word_label == 'dog': return [0,1]



def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)): #tqdm은 진행상태를 상태바에 나타내주는 부가적 기능을 하는 거라서 원하지 않으면 그냥 tqdm빼고 ()안의 내용 써주면 된다.
        label = label_img(img)              #위에서 만들어준 label 반환해주는 함수 이용
        path = os.path.join(TRAIN_DIR,img)  #Train_DIR/img 주소로 찾아갈 수 있게 지정
        img_data = cv2.imread(path,cv2.IMREAD_GRAYSCALE) #numpy.ndarray형태(행렬)로 저장된다.
                                                         #cv2.imread에 대한 설명: https://opencv-python.readthedocs.io/en/latest/doc/01.imageStart/imageStart.html
        img_data = cv2.resize(img_data, (IMG_SIZE,IMG_SIZE))        #50*50짜리로 resize
        training_data.append([np.array(img_data),np.array(label)])  #training_data 리스트에다가 이미지를 행렬로 바꿔준 거 추가
    shuffle(training_data)
    np.save('train_data.npy', training_data)  #나중에 이거 다시 돌리고 싶을 때 다시 돌릴 필요없고 load하면 되게 저장
    return training_data

def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img_data), img_num])
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data



# If dataset is not created:
train_data = create_train_data()
test_data = create_test_data()

# If you have already created the dataset:
# train_data = np.load('train_data.npy')
# test_data = np.load('test_data.npy')


############spliting the data ##############
# 24,500 images for training and 500 for testing(testing써서 overfitting 문제등을 피하기 위해서)
train = train_data[:-500]  #마지막 500개 제외하고 다 training data로
test = train_data[-500:]   #마지막 500개를 testing data로

# 데이터 읽어올 때 ([np.array(img_data), img_num])형태로 읽어온 거를 이제 feature data(사진에 대한 정보)와 cat,dog인지 알려주는 label로 나눠서 저장
X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = [i[1] for i in train]

X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = [i[1] for i in test]




import tensorflow as tf

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

tf.reset_default_graph()


convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
#input_data: 50*50*1


convnet = conv_2d(convnet, 32, 5, activation='relu')
#tflearn.layers.conv.conv_2d (incoming, nb_filter, filter_size, strides=1, padding='same', activation='linear',
#                             bias=True, weights_init='uniform_scaling', bias_init='zeros', regularizer=None, 
#                             weight_decay=0.001, trainable=True, restore=True, reuse=False, scope=None, name='Conv2D')
#                             http://tflearn.org/layers/conv/
        #input= convnet
        #필터= 5*5짜리 32개
        #stride=1 (디폴트)
        #padding='same' (디폴트)
        #activation function= relu
#tflearn.layers.conv.max_pool_2d (incoming, kernel_size, strides=None, padding='same', name='MaxPool2D')
#size=50*50*32
convnet = max_pool_2d(convnet, 5)
#tflearn.layers.conv.max_pool_2d (incoming, kernel_size, strides=None, padding='same', name='MaxPool2D')
        #input= convnet
        #필터= 5*5짜리
        #stride=NONE (Default: same as kernel_size.)
        #padding='same' (디폴트)
        #activation function= relu
#size=10*10*32


convnet = conv_2d(convnet, 64, 5, activation='relu')
#size=10*10*64
convnet = max_pool_2d(convnet, 5)
#size=2*2*64 


convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)


convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)


convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)


convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)


convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)


convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')


model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)


model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10, 
          validation_set=({'input': X_test}, {'targets': y_test}), 
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)



'''출력결과
Training Step: 3829  | total loss: 0.28708 | time: 73.099s
| Adam | epoch: 010 | loss: 0.28708 - acc: 0.8866 -- iter: 24448/24500
Training Step: 3830  | total loss: 0.28767 | time: 74.276s
| Adam | epoch: 010 | loss: 0.28767 - acc: 0.8854 | val_loss: 0.60003 - val_acc: 0.7840 -- iter: 24500/24500
'''


model.save(MODEL_NAME)



###이제 제대로 했는지 test data 몇개 돌려보자###
import matplotlib.pyplot as plt

# if you need to create the data:
#test_data = create_test_data()

# if you already have some saved:(우리는 위에서 test_data = create_test_data()해줬으니까)
test_data = np.load('test_data.npy')

fig=plt.figure() 

#test_data는 총 12,500개의 사진들이 [np.array(img_data), img_num] 형태로 변환되어서 저장되어 있다.
#밑의 for 문을 통해서 끝에서 24개가 각각 아래의 형태로 가져와진다.
# num   data
# 0     [np.array(img_data), img_num]
# 1     [np.array(img_data), img_num]
# ...
# 23    [np.array(img_data), img_num]

for num,data in enumerate(test_data[:24]):
    # cat: [1,0]
    # dog: [0,1]
    
    #위에서 create_test_data 함수 보면
    img_num = data[1]   #사진 파일 번호
    img_data = data[0]  #사진 파일 정보
    
    y = fig.add_subplot(3,8,num+1)  #결과 출력 형식 지정
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    model_out = model.predict([data])[0]  #data만 입력받아서 하나의 값만 출력하는 거라서 0th element이 우리의 관심사
    
    if np.argmax(model_out) == 1: str_label='Dog'
    else: str_label='Cat'
        
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    
    #x축, y축 눈금 제거해주기
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()


