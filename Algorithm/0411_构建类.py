
class MKAverage:
    def __init__(self, m: int, k: int):
        self.m=m
        self.k=k
        self.nums=[]

    def addElement(self, num: int) -> None:
        self.nums.append(num)

    def calculateMKAverage(self) -> int:
        if len(self.nums)<self.m:
            return -1
        keys=self.nums[-1*self.m:]
        t=self.k
        keys.sort()
        print(keys)
        return 1

if __name__=="__main__":
    obj = MKAverage(3, 1)
    obj.addElement(3)
    obj.addElement(1)
    t=obj.calculateMKAverage()
    print(t) 
    obj.addElement(10) 
    t=obj.calculateMKAverage()
    print(t) 
