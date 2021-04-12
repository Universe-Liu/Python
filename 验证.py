import numpy as np
import matplotlib.pyplot as plt
k1=[11.92204,11.86958,11.77726,11.70106,11.59425,11.54631,11.56847]
k2=[11.93946,11.90272,11.83625,11.77937,11.69340,11.64267,11.62787]
k3=[12.49739,12.43561,12.32689,12.16463,12.07198,12.04660,12.08885]
h=[0.05,0.1,0.2,0.3,0.5,0.7,0.95]
k1=np.array(k1)
k3=np.array(k3) 
h=np.array(h)
f1=np.polyfit(k1,h,3)
p1 = np.poly1d(f1)
print('p1 is :\n',p1)
f3=np.polyfit(k3,h,3)
p3 = np.poly1d(f3)
print('p3 is :\n',p3)
plot10045 = plt.plot(h, k1, 'r',label='2')
plot10226 = plt.plot(h, k3, 'b',label='5')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4) #指定legend的位置右下角
plt.title('polyfitting')
plt.show()