def getXORSum(arr1, arr2):
    andl=[]
    for i in range(len(arr1)):
        x=arr1[i]
        for j in range(len(arr2)):
            y=arr2[j]
            andl.append(x&y)
        
    if len(andl)==1:
        return andl[0]
    else:
        res=andl[0]
        for i in range(1,len(andl)):
            res=res^andl[i]
    return res

arr1 = [12]
arr2 = [4]
print(getXORSum(arr1,arr2))