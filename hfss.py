import matplotlib.pyplot as plt
import numpy as np

fre=[]                      #数据存放列表
db1=[]
db2=[]
db3=[]
db4=[]
i=0
file=open(r'data.txt', 'r')
data_tmp = file.readlines()  # 读取文件
for line in data_tmp:
    i=i+1
    if i>=8:                 #数据开始行
        data_read=line.strip().split(' ')
        fre.append(float(data_read[0]))
        db1.append(float(data_read[1]))
        db2.append(float(data_read[2]))
        db3.append(float(data_read[3]))
        db4.append(float(data_read[4]))

plt.plot(fre,db1)             #绘制图像
plt.show()