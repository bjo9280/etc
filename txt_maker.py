import os
from os.path import splitext


def txt_maker(path):
	for root, dirs, files in os.walk(path):
		for fname in files:
			string_ = splitext(fname)[0]
			fi.write(string_ + '\n')
			fo.write('/JPEGImages/' + string_ + '.jpg /SegmentationClassAug/' + string_ + '.png\n')

fi = open('train.txt', 'w')
fo = open('train2.txt', 'w')

path = 'D:/seg/JPEGImages/'
txt_maker(path)

fi.close()
fo.close()




