def findTheWinner(n, k) -> int:
    name=[]
    for i in range(n):
        name.append(i+1)
    cur=1
    while(len(name)>1):
        next_flag=(cur+k-2)%len(name)
        name.pop(next_flag)
        cur=(next_flag+1)%len(name)
    return name[0]
print(findTheWinner(5,2))