class SeatManager:
    seats=[]
    def __init__(self, n: int):
        for i in range(n):
            self.seats.append(i+1)

    def reserve(self) -> int:
        if len(self.seats):
            flag=self.seats[0]
            for i in range(1,len(self.seats)):
                if self.seats[i]<flag:
                    flag=self.seats[i]
        self.seats.remove(flag)
        return flag
        
    def unreserve(self, seatNumber: int) -> None:
        self.seats.append(seatNumber)


a=SeatManager(4)
print(a.reserve())
print(a.unreserve(1))
print(a.reserve())
print(a.reserve())  
print(a.reserve())
print(a.unreserve(2))     
print(a.reserve())   
print(a.unreserve(1))     
print(a.reserve())   
print(a.unreserve(2)) 
# Your SeatManager object will be instantiated and called as such:
# obj = SeatManager(n)
# param_1 = obj.reserve()
# obj.unreserve(seatNumber)