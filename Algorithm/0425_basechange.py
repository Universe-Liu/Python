def sumBase(n,k):
    ans=[]
    while(n):
        ans.append(n%k)
        n=int(n/k)
    return sum(ans)
print(sumBase(34,6))