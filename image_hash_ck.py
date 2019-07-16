# -*- coding: utf-8 -*-

from imutils import paths
import hashlib
import shutil
import os

file_list = list(paths.list_images('d:/images'))

def getHash(path, blocksize=65536):
    afile = open(path, 'rb')
    hasher = hashlib.md5()
    buf = afile.read(blocksize)
    while len(buf) > 0:
        hasher.update(buf)
        buf = afile.read(blocksize)
    afile.close()
    return hasher.hexdigest()

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

hash_list = []

for file in file_list:
    h = getHash(file)
    file.replace('\\', '/')
    if h in hash_list:
        # print(file)
        print('포함')
    else:
        print(file)
        new_file = file.replace('images', 'images/new')
        ensure_dir(new_file)
        print(new_file)
        shutil.copy(file, new_file)
    hash_list.append(h)
