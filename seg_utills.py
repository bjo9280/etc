from PIL import Image
import vis
from scipy.misc import imread
import os
import numpy as np
import re
import cv2

#change the name of output images when option is --also_save_raw_predictions=True
def dequote(fname):
    fname = os.path.splitext(fname)[0]
    parse = re.sub('[b]', '', fname) #remove b

    if(parse[0] == parse[-1]) and parse.startswith(("'",'"')):
        return parse[1:-1]
    return parse


def convert_grayscale_to_rgb(folder, num_classes):
    for root, dirs, files in os.walk(folder):
        for fname in files:
            im = imread(folder+fname)
            voc_palette = vis.make_palette(num_classes)
            out_im = Image.fromarray(vis.color_seg(im, voc_palette))
            print(dequote(fname))
            out_im.save("{}{}_prediction.png".format(folder, dequote(fname)))

def convert_rgb_to_grayscale(folder):
    for root, dirs, files in os.walk(folder + 'SegmentationClassPNG'):
        for fname in files:
            im = Image.open(folder + "SegmentationClassPNG/" + fname)
            in_ = np.array(im, dtype=np.float32)
            stacked_img = np.stack((in_,) * 3, -1)
            cv2.imwrite(folder + "SegmentationClassAug/{}".format(fname), stacked_img)

def txt_maker(path):
    fi = open('train.txt', 'w')
    fo = open('train2.txt', 'w')

    for root, dirs, files in os.walk(path):
        for fname in files:
            string_ = os.path.splitext(fname)[0]
            fi.write(string_ + '\n')
            fo.write('/JPEGImages/' + string_ + '.jpg /SegmentationClassAug/' + string_ + '.png\n')
    fi.close()
    fo.close()
