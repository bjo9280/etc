import sys
import os
import time
import re
import json
import numpy as np
import tensorflow as tf
import cv2

weights_file = 'frozen.pb'
label_file = 'Label_cls.txt'
cls_model_json = 'cls_model.json'

_default_input_name = 'input'
_default_output_name = 'resnet_v2_101/SpatialSqueeze'
_text_encoding = 'utf-8'

def load_graph(frozen_graph_filename):
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(frozen_graph_filename, 'rb') as f:
        graph_def.ParseFromString(f.read())
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def)  # , name='prefix')
    return graph

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

graph = load_graph(weights_file)

cls_model_obj = dict(
        input_name=_default_input_name,
        output_name=_default_output_name,
    )
cls_model_file = os.path.join(os.path.dirname(weights_file), cls_model_json)

with open(cls_model_file, encoding=_text_encoding) as f:
    cls_model_obj.update(json.load(f))

input_name = cls_model_obj['input_name']
output_name = cls_model_obj['output_name']

image_width = 224
image_height = 224

input_tensor = graph.get_tensor_by_name('import/' + input_name + ':0')
output_tensor = graph.get_tensor_by_name('import/' + output_name + ':0')


if 'vgg' in cls_model_obj['model_name']:
    with graph.as_default():
        output_tensor = tf.nn.softmax(output_tensor)
print((input_tensor.name, str(input_tensor.get_shape()), output_tensor.name, str(output_tensor.get_shape())))

config = tf.ConfigProto(gpu_options={'allow_growth': True})
sess = tf.Session(graph=graph, config=config)

det_settings = dict(
    sess=sess,
    input_tensor=input_tensor,
    output_tensor=output_tensor,
    image_width=image_width,
    image_height=image_height,
)

def txt_to_imglist(txt_file):
    lines = []
    img_list = []
    label_list = []
    with open(txt_file, encoding='utf-8-sig') as f:
        for line in f:
            lines.append([n.replace('\\', '/') for n in line.strip().split(' ')])
        for pair in lines:
            img_list.append((pair[0], pair[1]))
    return img_list

gt = []
y_pred = []

img_list = txt_to_imglist('Val_cls.txt')
testPaths = []
for i in img_list:
    testPaths.append(i[0])
    gt.append(i[1])

# testPaths = list(paths.list_images(path))

def detect_on_images(settings):
    sess = settings['sess']
    input_tensor = settings['input_tensor']
    output_tensor = settings['output_tensor']
    image_width = settings['image_width']
    image_height = settings['image_height']

    y_pred = []

    for file in testPaths:
        im = read_image(file, flags=cv2.IMREAD_COLOR)
        im = cv2.resize(im, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
        score = sess.run(output_tensor, feed_dict={input_tensor: im})
        class_id = np.argmax(score[0], axis=0)
        y_pred.append(int(class_id))
    return y_pred

y_pred = detect_on_images(det_settings)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

gt = list(map(int, gt))

confusion = confusion_matrix(gt, y_pred)
print('confusion matrix:')
print(confusion)
report = classification_report(gt, y_pred, output_dict=False)
print('classification_report:')
print(report)