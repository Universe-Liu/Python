def finds(strs,s):
    for i in range(len(strs)):
        if strs[i]==s:
            return i

def srea(str1,str2,f,z):
    flag[f]=1
    f=finds(str1,str2[f])
    if(flag[f]==0):
        srea(str1,str2,f,z)
    else:
        for j in range(n):
            if flag[j]==0:
                f=j
                z=z+1
                flag[0]=z
                srea(str1,str2,f,z)

if __name__=="__main__":
    n=int(input())
    if n>1:
        str1=[]
        str2=[]
        flag=[0]*n
        f=0
        res=1
        for i in range(n):
            st=input()
            strstr=st.split(" ")
            str1.append(strstr[0])
            str2.append(strstr[1])
        srea(str1,str2,f,res)
        print(flag)
    else:
        print(0)