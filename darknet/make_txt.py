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

sys.stdout = open('1117_test.txt', 'a')
open = ('./1117_test.txt', 'a')


# C:\Users\user\anaconda3\Lib\site-packages\darknet\data\1117

# main_file_path = 'C:/darknet-master/darknet-master/build/darknet/x64/data/1117/obj'

# main_file_path = 'C:/Users/user/anaconda3/Lib/site-packages/darknet/data/1117/obj'
main_file_path = 'C:/Users/user/anaconda3/Lib/site-packages/darknet/data/1117/test'

for m in os.listdir(main_file_path) :
    if m[-3:] == 'jpg' :
        # print('C:/Users/user/anaconda3/Lib/site-packages/darknet/data/1117/obj' + m)
        print('C:/Users/user/anaconda3/Lib/site-packages/darknet/data/1117/test' + m)