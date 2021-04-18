tasks =[[7,10],[7,12],[7,5],[7,4],[7,2]]
def getOrder(tasks):
    if len(tasks):
        ans=[]
        tasks_working=[]
        n=len(tasks)
        for i in range(n):
            tasks[i].append(i)
        tasks_waitwork=sorted(tasks,key=lambda x:[x[0],x[1],x[2]])
        t=tasks_waitwork[0][0]
        tasks_working.append(tasks_waitwork[0])
        tasks_waitwork=tasks_waitwork[1:]
        while n:
            t=t+tasks_working[0][1]
            ans.append(tasks_working[0][2])
            tasks_working=tasks_working[1:]
            n=n-1
            u=tasks_waitwork.copy()
            print(tasks_waitwork,tasks_working,ans)
            for i in range(len(tasks_waitwork)):
                if (tasks_waitwork[i][0]<=t) :
                    tasks_working.append(tasks_waitwork[i])
                    u.remove(tasks_waitwork[i])
                else:
                    break
            tasks_waitwork=u.copy()
            tasks_working=sorted(tasks_working,key=lambda x:[x[1],x[2]])
        return ans
    else:
        return []

a=getOrder(tasks)
print(a)