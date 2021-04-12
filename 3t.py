
def best_time(data,row,col,i,j,tm,res):
    tm=tm+data[i][j]
    if(tm<=t):
        if(i==row-1)&(j==col-1):
            res.append(tm)
        if(i<row-1) & (tm<t):
            best_time(data,row,col,i+1,j,tm,res)
        if(j<col-1) & (tm<t):
            best_time(data,row,col,i,j+1,tm,res)

if __name__=="__main__":
    setting=input().split(" ")
    row,col,t=int(setting[0]),int(setting[1]),int(setting[2])
    if (row>0) & (col>0):
        data=[]
        for i in range(row):
            t1=input().split(" ")
            tmp=[]
            for j in range(col):
                tmp.append(int(t1[j]))
            data.append(tmp)
        res=[]
        best_time(data,row,col,0,0,0,res)
        if res!=[]:
            print(max(res))
        else:
            print(-1)