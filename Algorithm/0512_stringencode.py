shuru = input()

a = {}
inputflag = 0
zimu = ''
shuzi = ''

for i in shuru:
    if ord('A') <= ord(i) <= ord('Z') or ord('a') <= ord(i) <= ord('z'):
        if inputflag == 1:
            a[zimu] = int(shuzi)
            zimu  = ''
            shuzi = ''
            inputflag = 0
        zimu += i

    else:
        shuzi += i
        inputflag = 1

a[zimu] = int(shuzi) #最后一组字典获取
listshuru = list(a.items()) #字典转成列表，列表元素是元祖


for i in range(len(listshuru)):
    for j in range(i+1, len(listshuru)):
        if i+1 == len(listshuru):
            break
        if listshuru[i][1] > listshuru[j][1]:
            listshuru[i],listshuru[j] = listshuru[j],listshuru[i]
        elif listshuru[i][1] == listshuru[j][1]:
            if listshuru[i][0] > listshuru[j][0]:
                listshuru[i],listshuru[j] = listshuru[j],listshuru[i]
for i in listshuru:
    print('{}'.format(i[0]*i[1]), end = '')