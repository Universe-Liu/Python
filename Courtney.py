import math
from sympy import *
from scipy import special
from scipy.optimize import fsolve
c=3*math.pow(10,8)
ll=[18.26,16.5,17.78,18.54,14.76]
h=[8.57,7.67,7.55,8.63,6.57]
f=[2.502,2.635,2.546,2.661,2.925]
Qu=[3941,1955,1978,3707,1997]
li=[1,4,5,7,'#']

ll2=[14.76,15.24,14.58,14.92,14.8]
h2=[6.57,7.19,5.93,6.39,7.05]
f2=[2.925,2.833,2.961,2.893,2.917]
def get_anotherv(d,L,f0):
    v1=math.pow(0.5*math.pi*d/L,2)
    v2=math.pow(d*math.pi*f0/c,2)
    return math.sqrt(v1-v2)

def get_v(d,L,f0):
    v1=math.pow(math.pi*d*f0/c,2)
    v2=math.pow(c/(2*f0*L),2)-1
    return math.sqrt(v1*v2)

def get_u(v):
    def fun_u(i,v):
        u=i[0]
        return u*special.j0(u)*special.k1(v)+v*special.k0(v)*special.j1(u)
    u=fsolve(fun_u,[3],v)[0]
    print(u,v)
    return u

def get_Xi(d,L,f0):
    v=get_v(d,L,f0)
    u=get_u(v)
    xi1=math.pow(c,2)/math.pow(math.pi*d*f0,2)
    xi2=v*v+u*u
    return xi1*xi2+1

def get_Rs(f0):
    return math.sqrt(math.pi*f0*4*math.pi*10**(-14)/5.8)

def get_W(u,v):
    w1=special.j1(u)/special.k1(v)
    w2=special.k0(v)*special.kn(2,v)-special.k1(v)*special.k1(v)
    w3=special.j1(u)*special.j1(u)-special.j0(u)*special.jv(2,u)
    return math.pow(w1,2)*w2/w3

def get_B(L,f0,xi,W,rs):
    b1=math.pow(c/(f0*2*L),3)
    b2=rs*(1+W)/(30*math.pow(math.pi,2)*xi)
    return b1*b2

def get_A(xi,W):
    return 1+W/xi

def get_sigma(f0,rs):
    return 0.00825*f0/rs

def get_delta(d,L,f0,xi,Q):
    v=get_v(d,L,f0)
    u=get_u(v)
    W=get_W(u,v)
    rs=get_Rs(f0)
    delta1=get_A(xi,W)/Q
    delta2=get_B(L,f0,xi,W,rs)/math.sqrt(get_sigma(f0,rs))
    return delta1-delta2

for i in range(len(ll)):
    xi=get_Xi(ll[i]*10**(-3),h[i]*10**(-3),f[i]*10**9)
    delta=get_delta(ll[i]*10**(-3),h[i]*10**(-3),f[i]*10**9,xi,Qu[i])
    print(li[i],xi,delta)

for i in range(len(ll2)):
    xi=get_Xi(ll2[i]*10**(-3),h2[i]*10**(-3),f2[i]*10**9)
    #delta=get_delta(ll[i]*10**(-3),h[i]*10**(-3),f[i]*10**9,xi,Qu[i])
    print(xi)