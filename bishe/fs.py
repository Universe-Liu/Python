h=[0.05,0.1,0.2,0.3,0.5,0.7,0.95]
xi=[2,5,10,20,35,50,75,100,150]
k=[]

with open('kxi.txt') as f:
    fs_str=f.readlines()
    for i in range(len(fs_str)):
        fs_s=fs_str[i].strip().split('\t')
        k_t=[]
        for j in range(len(fs_s)):
            if j:
                k_t.append(float(fs_s[j]))
        k.append(k_t)
for i in range(len(xi)):
    fs=[]
    for j in range(len(h)):
        fs.append(10-10*h[j]*(xi[i]-1)*k[i][j]/1000)
        #print(h[j],xi[i])
    print(fs)