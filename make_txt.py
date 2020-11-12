import os
import sys

# main_file_path = 'D:/watcher/videos/videos'
# main_save_path = 'D:/watcher/videos/images'

# count = 0

# for m in os.listdir(main_file_path):
#     for n in os.listdir(main_file_path + '/' + m):
#         file_path = main_file_path + '/' + m + '/' + n
#         file_list = os.listdir(file_path) 

#         print(file_list)

sys.stdout = open('580_cone_test.txt', 'a')
open = ('./580_cone_test.txt', 'a')

main_file_path = 'C:/darknet-master/darknet-master/build/darknet/x64/data/580_cone/dataset/test'

for m in os.listdir(main_file_path) :
    if m[-3:] == 'jpg' :
        print('C:/darknet-master/darknet-master/build/darknet/x64/data/580_cone/dataset/test/' + m)