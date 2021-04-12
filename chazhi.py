import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d



X1 = np.array([0.05,0.1,0.2,0.4,0.7,0.8,1])
Y1 = np.array([2,5,10,15,20])
Z1 = [[12.497,12.436,12.327,12.165,12.072,12.047,12.089],[12.512,12.463,12.374,12.232,12.135,12.081,12.068 ],
    [12.535,12.505,12.442,12.310,12.175,12.039,11.903],[12.535,12.505,12.442,12.310,12.175,12.039,11.903],
    [12.579,12.580,12.541,12.342,12.027,11.636,11.202]]

#去掉第三列值
X2 = np.array([0.05,0.1,0.4,0.7,0.8,1])
Y2 = np.array([2,5,10,15,20])
Z2 = [[12.497,12.436,12.165,12.072,12.047,12.089],[12.512,12.463,12.232,12.135,12.081,12.068 ],
    [12.535,12.505,12.310,12.175,12.039,11.903],[12.535,12.505,12.310,12.175,12.039,11.903],
    [12.579,12.580,12.342,12.027,11.636,11.202]]


f1 = interp2d(X1,Y1,Z1)
f2 = interp2d(X2,Y2,Z2)
print(f1(0.2,4.032699))
print(f1(0.2,4.032699))