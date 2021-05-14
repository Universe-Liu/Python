def replaceDigits(s):
    sl=list(s)
    i = 1
    for i in range(1,len(sl)):
        pre=sl[i-1]
        if(i % 2!=0):
            k = int(sl[i])
            t=chr(ord(pre)+k)
            sl[i] = t
        i = i+1
    return "".join(sl)

print(replaceDigits("a1c1e1"))
