costs=[1,6,3,1,2,5]
coins=20

def maxIceCream(costs, coins) -> int:
    ans=0
    s=0
    costs.sort()
    if costs[0]>coins:
        return ans
    for i in range(len(costs)):
        if costs[i]+s<=coins:
            ans+=1
            s+=costs[i]
        else:
            break
    return ans

print(maxIceCream(costs,coins))