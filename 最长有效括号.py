def longestV(s):
    res = 0
    max_i=0
    l = []
    ind = []
    n = len(s)
    ss = []
    if(n > 1):
        i = s.index('(')
        for j in range(i, n):
            ss.append(s[j])
            ind.append(j)
            if(len(ss) > 1):
                if(ss[-1] == ')') and (ss[-2] == '('):
                    l.append(ind[-2])
                    ind = ind[:-2]
                    ss = ss[:-2]
        if(len(l) > 2):
            l.sort()
            print(l)
            for i in range(0, len(l)):
                res=2
                for j in range(1, len(l)-i):
                    if(l[i+j]-l[i] < 3):
                        res = res+2
                print("%d :%d"% (l[i],max_i))
                if(res>max_i):
                    max_i=res
            res=max_i
        elif(len(l)==1):
            res=2
        elif(len(l)==2):
            res=2
            if(l[1]-l[0]<3):
                res=res+2
    return res


a ="()(())"
b=")()())()()("


print(longestV(b))