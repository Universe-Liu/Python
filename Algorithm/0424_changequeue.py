'''
输入描述:
第一行三个以空格分隔的整数n1,n2,q
第二行q个以空格分隔的整数，表示离开队伍的编号。
1≤n1,n2,q ≤10^5
保证离开队伍的人员编号在[1,n1+n2]范围内。

输出描述:
共两行整数，分别描述两个队伍的样子，按照离收银台近的位置开始依次给出人员的编号（以空格分隔）。
保证最后两个队伍均至少有一个人。
'''
l = input().split(' ')
n1, n2, q = int(l[0]), int(l[1]), int(l[2])
qt = {}
t = list(map(int, input().split()))
for i in range(n1):
    qt[i+1] = 1
for i in range(n2):
    qt[i+n1+1] = 2
for i in t:
    if qt[i] == 1:
        del qt[i]
        qt[i] = 2
    else:
        del qt[i]
        qt[i] = 1
for i, j in qt.items():
    if j == 1:
        print(i, end=' ')
print()
for i, j in qt.items():
    if j == 2:
        print(i, end=' ')
