import sys
import os
import time
import re

import numpy as np
import tensorflow as tf
import cv2

_slimpath = os.path.join('../slim')
sys.path.insert(0,_slimpath)

from nets.nets_factory import get_network_fn
from preprocessing.preprocessing_factory import get_preprocessing

_model_name = 'vgg_16'
_input_name = 'input'
_image_size = 224
_dataset_num_classes = 4 # len(class_names)
_preprocessing_name = 'vgg'
_feature_endpoint = 'vgg_16/fc7'

input_checkpoint = 'save/model.ckpt-10000'



def read_image(im_path,flags=cv2.IMREAD_UNCHANGED):
    if im_path.endswith('.npy'):
        im = np.load(im_path)
    else:
        with open(im_path,'rb') as f:
            im_encoded = f.read()
        im = cv2.imdecode(np.frombuffer(im_encoded,dtype=np.uint8),flags=flags)
        if len(im.shape) > 2:
            im = im[:,:,::-1]
    return im

# 네트워크 모델 구성
network_fn = get_network_fn(
    _model_name,
    num_classes=_dataset_num_classes,
    is_training=False
)

# 텐서플로우 내부 이미지 전처리 구성
preprocessing_fn = get_preprocessing(_preprocessing_name,is_training=False)
image_input = tf.placeholder(name=_input_name,shape=[None,None,3],dtype=tf.uint8)
preprocessed = preprocessing_fn(image_input, _image_size, _image_size)
logits, endpoint = network_fn(tf.expand_dims(preprocessed,axis=0))
feature_tensor = endpoint[_feature_endpoint]
_feature_size = feature_tensor.get_shape().as_list()[-1]

# 세션 생성 및 학습모델 로딩
config = tf.ConfigProto(gpu_options={'allow_growth':True})
sess = tf.Session(config=config)
sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()
saver.restore(sess, input_checkpoint)

def txt_to_imglist(txt_file):
    lines = []
    img_list = []
    with open(txt_file, encoding='utf-8-sig') as f:
        for line in f:
            lines.append([n.replace('\\', '/') for n in line.strip().split(' ')])
        for pair in lines:
            img_list.append((pair[0], pair[1]))
    return img_list

def predict(im_name,preproc_fn=None):
    preds_, feature_tensor = sess.run([logits, endpoint[_feature_endpoint]], {image_input: im_name})
    y = np.argmax(preds_[0])
    score = tf.nn.softmax(preds_[0])[y]
    return y, score, feature_tensor

test_gt = []
y_pred = []

val_list = txt_to_imglist('Val_cls.txt')

val_features = np.empty((len(val_list), 1 + _feature_size), dtype=np.float32)

testPaths = []
for i, img in enumerate(val_list):
    testPaths.append(img[0])
    test_gt.append(img[1])

for i, path in enumerate(testPaths):
    im = read_image(path, flags=cv2.IMREAD_COLOR)
    im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_LINEAR)
    y, score, features = predict(im)

    val_features[i:i + 1, :_feature_size] = features.reshape((-1, _feature_size))
    val_features[i:i + 1, -1] = test_gt[i]

    print(path, y, test_gt[i])

train_gt = []
y_pred = []

train_list = txt_to_imglist('Train_cls.txt')

train_features = np.empty((len(train_list), 1 + _feature_size), dtype=np.float32)

trainPaths = []
for i, img in enumerate(train_list):
    trainPaths.append(img[0])
    train_gt.append(img[1])

for i, path in enumerate(trainPaths):
    im = read_image(path, flags=cv2.IMREAD_COLOR)
    im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_LINEAR)
    y, score, features = predict(im)

    train_features[i:i + 1, :_feature_size] = features.reshape((-1, _feature_size))
    train_features[i:i + 1, -1] = train_gt[i]

    print(path, y, train_gt[i])

X_test = val_features[:,:-1].reshape([-1,4096])
y_test  = val_features[:,-1:].reshape([-1,1])

X_train = train_features[:,:-1].reshape([-1,4096])
y_train  = train_features[:,-1:].reshape([-1,1])

from xgboost import XGBClassifier

model_name = 'XGB'
print('training {:s} on training features...'.format(model_name))
model = XGBClassifier(tree_method='gpu_hist',verbosity=True,verbose=3)
print(model)

model.fit(X_train,y_train,verbose=3)
pred = model.predict(X_test)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

gt = list(map(int, test_gt))
y_pred = list(map(int, pred))

confusion = confusion_matrix(gt, y_pred)
print('confusion matrix:')
print(confusion)
report = classification_report(gt, y_pred, output_dict=False)
print('classification_report:')
print(report)