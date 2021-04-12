'''
给定 N 个无限容量且初始均空的水缸，每个水缸配有一个水桶用来打水，第 i 个水缸配备的水桶容量记作 bucket[i]。小扣有以下两种操作：

升级水桶：选择任意一个水桶，使其容量增加为 bucket[i]+1
蓄水：将全部水桶接满水，倒入各自对应的水缸
每个水缸对应最低蓄水量记作 vat[i]，返回小扣至少需要多少次操作可以完成所有水缸蓄水要求。

注意：实际蓄水量 达到或超过 最低蓄水量，即完成蓄水要求。
'''
from math import ceil
def storeWater(bucket,vat) -> int:
    key=[]
    pu=[]
    for i in range(len(vat)):
        divd=[]
        if(vat[i]):
            for j in range(bucket[i],vat[i]):
                if(bucket[i]):
                    divd.append(ceil(float(vat[i])/float(j)))
                else:
                    bucket[i]=1
                    divd.append(vat[i]+1)
            pu.append(divd)
    temp=0
    if(len(pu)==1):
        return pu[0][0]
    for i in range(len(pu[0])):
        for j in range(1,len(pu)):
            if(pu[0][i] in pu[j]):
                temp+=pu[j].index(pu[0][i])+pu[0][i]+i
                if j==len(pu)-1:
                    key.append(temp)
            else:
                break
    return min(key)
b=[9,0,1]
v=[0,0,2]
print(storeWater(b,v))