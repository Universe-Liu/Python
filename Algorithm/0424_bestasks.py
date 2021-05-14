def findmax(q,p):
    if 0 not in p:
        ed=' '
        for i in range(5):
            if i==4:
                ed='\n'
            print(q[i][-1],end=ed)
            q[i].pop()
    else:
        print("-1",end='\n')

q=[[] for _ in range(5)]
n=int(input())
tasks=input().split(' ')
for i in range(n):
    tasks[i]=int(tasks[i])
    q[tasks[i]-1].append(i+1)
    p=[len(q[0]),len(q[1]),len(q[2]),len(q[3]),len(q[4])]
    findmax(q,p)