def maxFrequency(nums, k):
    nums.sort()
    n=len(nums)
    dif=[]
    for i in range(1,n):
        dif.append(nums[i]-nums[i-1])
    ans=0
    sumdif=0
    for i in range(len(dif)):
        dans=0
        for j in range(i,len(dif)):
            sumdif+=dif[j-i]*(j-i+1)
            if sumdif<=k:
                dans=dans+1
        print(dans)
        ans=max(dans,ans)
    return ans
nums=[1,4,8,13]
print(maxFrequency(nums,3056))