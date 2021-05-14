inputs=input().split(' ')
n,m=int(inputs[0]),int(inputs[1])
robotmap=[]

for i in range(n):
    s=input()
    tmp=[]
    for j in range(m):
        if s[j]=='.':
            tmp.append(0)
        elif s[j]=='*':
            tmp.append(1)
    robotmap.append(tmp)

def finipath(x,y,en,f,ener,visit):
    if (x==n-1) & (y==m-1):
        ener.append(en)
    else:
        if visit[x][y]!=1:
            if robotmap[x][y]==1:
                if f==1:
                    f=0
                    visit[x][y]=1
                    if x<n-1:
                        finipath(x+1,y,en+1,f,ener,visit)
                    if y>0:
                        finipath(x,y-1,en+1,f,ener,visit)
                    if y<m-1:
                        finipath(x,y+1,en+1,f,ener,visit)
                    visit[x][y]=0
                else:
                    return 
            else:
                visit[x][y]=1
                if x<n-1:
                    finipath(x+1,y,en+1,f,ener,visit)
                if y>0:
                    finipath(x,y-1,en+1,f,ener,visit)
                if y<m-1:
                    finipath(x,y+1,en+1,f,ener,visit)
                visit[x][y]=0
        else:
            return 
        
print(robotmap)
def findminienergy():
    visit=[[0 for _ in range(m)] for _ in range(n)]
    ener=[]
    finipath(0,0,0,1,ener,visit)
    if len(ener):
        return min(ener)
    else:
        return -1

print(findminienergy())