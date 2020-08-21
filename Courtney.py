import math
from sympy import *
from scipy import special
from scipy.optimize import fsolve
c=3*math.pow(10,11)
def get_v(d,L,f0):
    v1=math.pow(math.pi*d*f0,2)/math.pow(c,2)
    v2=0.5*c/(f0*L)-1
    return math.sqrt(v1*v2)

def get_u(v):
    def fun_u(i,v):
        u=i[0]
        return u*special.j0(u)+v*special.y0(v)*special.j1(u)/special.y1(v)
    u=fsolve(fun_u,[4],v)[0]
    print(u,v)
    return u

def get_Xi(d,L,f0):
    v=get_v(d,L,f0)
    u=get_u(v)
    xi1=math.pow(c,2)/math.pow(math.pi*d*f0,2)
    xi2=v*v+u*u
    return xi1*xi2+1

print(get_Xi(14.76,6.57,2.925*10**9))
