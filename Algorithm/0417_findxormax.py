nums=[0,1,1,3]

maximumBit = 2
'''
def nums_maximum(nums,maximumBit):
    ans=[]
    db=[[] * 1 for _ in range(maximumBit)]
    for i in range(len(nums)):
        t=nums[i]
        for j in range(maximumBit):
            db[j].append(t%2)
            t=int(t/2)
    def find(db,Bit,ans):
        res=0
        for i in range(maximumBit-1,-1,-1):
            res=res*2
            if db[i].count(1)%2==0:
                res=res+1
        ans.append(res)
    for i in range(len(nums)):
        find(db,maximumBit,ans)
        for j in range(maximumBit):
            db[j]=db[j][:-1]
    return ans

def nums_maximum(nums,maximumBit):
    ans=[]
    val=0
    for i in range(len(nums)):
        val=val^nums[i]
    while(len(nums)):
        res=0
        for j in range(maximumBit):
            if (val & (1<<j))==0:
                res=res^(1<<j)
        ans.append(res)
        val=val^nums[-1]
        nums=nums[:-1]
    return ans
'''
def nums_maximum(nums,maximumBit):
    max_num = 2 ** maximumBit - 1
    tmp = nums[0]
    ret = [max_num - tmp, ]
    for num in nums[1:]:
        ret.append(max_num - tmp ^ num)
        tmp = tmp ^ num
    return ret[::-1]
print(nums_maximum(nums,maximumBit))