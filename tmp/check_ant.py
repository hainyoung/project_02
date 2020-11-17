import os

# path = 'H:/hainyoung/yolov4_3classes/cone/obj/'
# path = 'H:/hainyoung/yolov4_3classes/cone/test/'
# path = 'H:/hainyoung/yolov4_3classes/bollard/obj/'
# path = 'H:/hainyoung/yolov4_3classes/bollard/test/'
# path = 'H:/hainyoung/yolov4_3classes/barrel/obj/'
path = 'H:/hainyoung/yolov4_3classes/barrel/test/'


# out = './tmp/result/cone_obj_ant.txt'
# out = './tmp/result/cone_test_ant.txt'
# out = './tmp/result/bollard_obj_ant.txt'
# out = './tmp/result/bollard_test_ant.txt'
# out = './tmp/result/barrel_obj_ant.txt'
out = './tmp/result/barrel_test_ant.txt'

out_file = open(out, 'w')

files = os.listdir(path)

for filename in files :
    if ".txt" not in filename :
        continue
    file = open(path + filename)
    for line in file:
        out_file.write(line)
    out_file.write('')
    file.close()
out_file.close()

# myfile = open('./tmp/result/cone_obj_ant.txt') # 1108
# myfile = open('./tmp/result/cone_test_ant.txt') # 174
# myfile = open('./tmp/result/bollard_obj_ant.txt') # 1145
# myfile = open('./tmp/result/bollard_test_ant.txt') # 377
# myfile = open('./tmp/result/barrel_obj_ant.txt') # 709
myfile = open('./tmp/result/barrel_test_ant.txt') # 238

print(len(myfile.readlines()))

myfile.close()