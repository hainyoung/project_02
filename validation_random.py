import random

txt = open('./traffic_cone_gen.txt','r')
f = open('./cone_gen_random.txt','w')

tmp = []

while True :
    line = txt.readline()
    if not line:
        break
        
    tmp.append(line)
    
random.shuffle(tmp)
        
for i in tmp :  
    f.write(i)

txt.close()
f.close()