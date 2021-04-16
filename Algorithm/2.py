s=input().split(' ')
n,m=int(s[0]),int(s[1])
lu=[]
for i in range(n):
    tmp=[]
    for j in range(n):
        if j>i:
            tmp.append(1)
        else:
            tmp.append(0)
    lu.append(tmp)
for i in range(m):
    s=input().split(' ')
    if s[0]>s[1]:
        s[0],s[1]=s[1],s[0]
    lu[int(s[0])-1][int(s[1])-1]=0
p=[1 for x in range(n)]
res=[]
t=[]
def find_lugroup(lu,res,t,i):
    if i not in t:
        t.append(i)
        p[i]=0
        for j in range(i,n):
            if lu[i][j]:
                find_lugroup(lu,res,t,j)
        for k in range(n):
            if p[k]:
                res.append(t)
                t=[]
                find_lugroup(lu,res,t,k)
key=[]
find_lugroup(lu,res,t,0)
print(len(res))
for i in range(len(res)):
    key.append(len(res[i]))
key.sort()
for i in range(len(key)-1):
    print(key[i],end=' ')
print(key[len(key)-1])