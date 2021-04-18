def checkIfPangram( sentence: str):
    if len(sentence)<26:
        return False
    else:
        v=[0]*26
        for s in sentence:
            t=ord(s)-ord("a")
            v[t]=v[t]+1
            if v.count(0)==0:
                return True
        return False
print(checkIfPangram("thequickbrownfoxjumpsoverthelazydog"))