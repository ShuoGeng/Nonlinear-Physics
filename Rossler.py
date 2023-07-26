# The file performs calculations and displays bifurcation diagrams of the Rossler system and Lyapunov exponents. 
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from math import sqrt,log,e
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
up = 10000
N = 10000
b0 = np.linspace(0,5,N)
# bx = np.zeros([up,N])
# for i in range(N):
#     bx[i,:] = b0.copy()
time = 10000
x0 = np.zeros([up,N])-1
x = np.zeros(max(up,time))
y = np.zeros(max(up,time))
z = np.zeros(max(up,time))

x[0] = 0.1
y[0] = 0.1
dt = 0.001
a = 0.2
c = 9
T = np.zeros([up,3,3])
T[:,0,0] = 1
T[:,1,1] = 1
T[:,2,2] = 1
mod0 = np.zeros([N,up])
xlist = np.zeros([N,4])
for j in range(N):
	b = b0[j]
	k = 0
	for i in range(time-1):
		k11 = dt*(-y[i]-z[i])
		k12 = dt*(x[i]+a*y[i])
		k13 = dt*(b+(x[i]-c)*z[i])
		
		k21 = dt*(-(y[i]+1/2*k12)-(z[i]+1/2*k13))
		k22 = dt*((x[i]+1/2*k11)+a*(y[i]+1/2*k12))
		k23 = dt*(b+((x[i]+1/2*k11)-c)*(z[i]+1/2*k13))
		
		k31 = dt*(-(y[i]+1/2*k22)-(z[i]+1/2*k23))
		k32 = dt*((x[i]+1/2*k21)+a*(y[i]+1/2*k22))
		k33 = dt*(b+((x[i]+1/2*k21)-c)*(z[i]+1/2*k23))
		
		k41 = dt*(-(y[i]+k32)-(z[i]+k33))
		k42 = dt*((x[i]+k31)+a*(y[i]+k32))
		k43 = dt*(b+((x[i]+k31)-c)*(z[i]+k33))
		
		x[i+1] = x[i] + 1/6*(k11+2*k21+2*k31+k41)
		y[i+1] = y[i] + 1/6*(k12+2*k22+2*k32+k42)
		z[i+1] = z[i] + 1/6*(k13+2*k23+2*k33+k43)
		
		x[0] = x[time-1]
		y[0] = y[time-1]
		z[0] = z[time-1]
	for i in range(up-1):   
		k11 = dt*(-y[i]-z[i])
		k12 = dt*(x[i]+a*y[i])
		k13 = dt*(b+(x[i]-c)*z[i])
		
		k21 = dt*(-(y[i]+1/2*k12)-(z[i]+1/2*k13))
		k22 = dt*((x[i]+1/2*k11)+a*(y[i]+1/2*k12))
		k23 = dt*(b+((x[i]+1/2*k11)-c)*(z[i]+1/2*k13))
		
		k31 = dt*(-(y[i]+1/2*k22)-(z[i]+1/2*k23))
		k32 = dt*((x[i]+1/2*k21)+a*(y[i]+1/2*k22))
		k33 = dt*(b+((x[i]+1/2*k21)-c)*(z[i]+1/2*k23))
		
		k41 = dt*(-(y[i]+k32)-(z[i]+k33))
		k42 = dt*((x[i]+k31)+a*(y[i]+k32))
		k43 = dt*(b+((x[i]+k31)-c)*(z[i]+k33))
		
		x[i+1] = x[i] + 1/6*(k11+2*k21+2*k31+k41)
		y[i+1] = y[i] + 1/6*(k12+2*k22+2*k32+k42)
		z[i+1] = z[i] + 1/6*(k13+2*k23+2*k33+k43)
		T[i+1][0][0] = T[i][0][0]+(-1*T[i][1][0]-1*T[i][2][0])*dt
		T[i+1][0][1] = T[i][0][1]+(-1*T[i][1][1]-1*T[i][2][1])*dt
		T[i+1][0][2] = T[i][0][2]+(-1*T[i][1][2]-1*T[i][2][2])*dt
		
		T[i+1][1][0] = T[i][1][0]+(1*T[i][0][0]+a*T[i][1][0])*dt
		T[i+1][1][1] = T[i][1][1]+(1*T[i][0][1]+a*T[i][1][1])*dt
		T[i+1][1][2] = T[i][1][2]+(1*T[i][0][2]+a*T[i][1][2])*dt
		
		T[i+1][2][0] = T[i][2][0]+(z[i]*T[i][0][0]+(x[i]-c)*T[i][2][0])*dt
		T[i+1][2][1] = T[i][2][1]+(z[i]*T[i][0][1]+(x[i]-c)*T[i][2][1])*dt
		T[i+1][2][2] = T[i][2][2]+(z[i]*T[i][0][2]+(x[i]-c)*T[i][2][2])*dt
		
		mod = (T[i+1][0][0]**2+T[i+1][0][1]**2+T[i+1][0][2]**2+\
		T[i+1][1][0]**2+T[i+1][1][1]**2+T[i+1][1][2]**2+\
		T[i+1][2][0]**2+T[i+1][2][1]**2+T[i+1][2][2]**2)
		mod0[j,i] = sqrt(mod)
		T[i+1][0][0] = T[i+1][0][0]/mod
		T[i+1][0][1] = T[i+1][0][1]/mod
		T[i+1][0][2] = T[i+1][0][2]/mod
		T[i+1][1][0] = T[i+1][1][0]/mod
		T[i+1][1][1] = T[i+1][1][1]/mod
		T[i+1][1][2] = T[i+1][1][2]/mod
		T[i+1][2][0] = T[i+1][2][0]/mod
		T[i+1][2][1] = T[i+1][2][1]/mod
		T[i+1][2][2] = T[i+1][2][2]/mod
		
	x2 = np.zeros(up)
	for i in range(up):
		x2[i] = np.log(np.abs(np.real(np.linalg.eig(T[i,:,:])[0][0])))/((i+1)*dt)
	if x2[i] < -10:
		x2[i] = 0
		blist[j] =np.log(np.abs(np.real(np.linalg.eig(T[up-1,:,:])\
		[0][0])))/(up*dt)
	meanlist = np.zeros(up)
	k = up-1
	for i in range(up):
		meanlist[i] = np.real(np.linalg.eig(T[i][:][:])[0][0])      
	k = up-1
	a11 = np.log(np.abs(np.real(np.linalg.eig(T[k][:][:])[0][0])))
	a21 = np.sum(np.log(mod0[j,0:k]))
	xlist[j,1] = (a11+a21)/(k*dt)
	k = k-1
	a12 = np.log(np.abs(np.real(np.linalg.eig(T[k][:][:])[0][0])))
	a22 = np.sum(np.log(mod0[j,0:k]))
	xlist[j,2] = (a12+a22)/(k*dt)
	xlist[j,3] = (a11+a12+a21+a22)/2/(k*dt)
fig = plt.figure()
plt.plot(x[int(up/2):up],y[int(up/2):up])
fig = plt.figure()
ax1 = plt.axes(projection='3d')
ax1.scatter3D(x[down:up],y[down:up],z[down:up],'gray')
b0xs = pd.DataFrame(b0)
b0xs.to_csv('b0xs.csv')
xlists = pd.DataFrame(xlist)
xlists.to_csv('xlist.csv')
fig = plt.figure()
ax = plt.plot(1,1,1)
plt.plot(b0,xlist*5+10)
plt.plot(b0,x0)
ax.set_xlim(0,5)
