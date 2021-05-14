def isbeautiful(word,start,stop):
    print(word[start:stop])
    if ('a' in word[start:stop])&('e' in word[start:stop])&('i' in word[start:stop])&('o' in word[start:stop])&('u' in word[start:stop]):
        return True
    else:
        return False


def longestBeautifulSubstring(word):
    ans=0
    start=0
    if len(word)!=1:
        l=0
        for i in range(1,len(word)):
            if i==len(word)-1:
                if isbeautiful(word,start,i+1):
                    ans=max(ans,i+1-start)
            if word[i-1]>word[i]:
                if isbeautiful(word,start,i):
                    ans=max(ans,i-start)
                start=i
    return ans
print(longestBeautifulSubstring("aeeeiiiioooauuuaeiou"))