# file = open('./txt/2020_11_16_16_45_14.txt', 'r')
# for line in file:
#     print(line.strip())

# import os
# folderpath = r"./txt"
# filepaths = [os.path.join(folderpath, name) for name in os.listdir(folderpath)]

# all_files = []

# for path in filepaths:
#     with open(path, 'r') as f:
#         file = f.readlines()
#         all_files.append(file)

# print(all_files)

import glob, os, shutil, sys, re

# sys.stdout = open('tmp_3.txt', 'w')

path = 'H:/hainyoung/1116_srt/txt'

def tmp():
    filenames = []
    files = os.listdir(path)

    # print(files)
    
    for file in files:
        # print(file)
        txtlines = open(path+'/'+file, 'r')
        for line in txtlines:
            # line = ''.join(line.split())
            # print(line)
            pattern = re.compile(r'\s+')
            line = re.sub(pattern, '', line)
            # print(line)
            # print(len(line))
            # if len(line) == 0:
            #     pass
            # elif len(line) != 0:
            #     print(line)
            # line = ','.join(line())
            print(line)            


if __name__ == '__main__' : tmp()