import math
dp=[0]*100000000
dp[2]=1

def isprime(k):
    flag=1
    m=int(math.sqrt(k))
    for i in range(2,m+1):
        if k%i==0:
            flag=0
            break
    if flag:
        return True
    else:
        return False

n=int(input())
if n>=3:
    for i in range(3,n+1):
        if isprime(i):
            dp[i]=dp[i-1]+1
        else:
            dp[i]=dp[i-1]
    print(dp[n])

