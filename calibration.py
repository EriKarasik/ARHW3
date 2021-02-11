from numpy import sin,cos,diag,arctan2,transpose,zeros,pi,outer,divide,linspace,stack,ones
from numpy import hstack as h,vstack as v
from numpy.linalg import multi_dot as dot,inv
from numpy.random import uniform, normal
import matplotlib.pyplot as plt

l1, l3= 0.4, 0.1
theta = [0, 0, 0]
kTheta = diag([1e+6, 2e+6, 0.5e+6])

def Rx(q): return [[1,0,0,0],[0,cos(q),-sin(q),0],[0,sin(q),cos(q),0],[0,0,0,1]]
def dRx(q):return [[0,0,0,0],[0,-sin(q),-cos(q),0],[0,cos(q),-sin(q),0],[0,0,0,0]]
def Ry(q): return [[cos(q),0,sin(q),0],[0,1,0,0],[-sin(q),0,cos(q),0],[0,0,0,1]]
def dRy(q):return [[-sin(q),0,cos(q),0],[0,0,0,0],[-cos(q),0,-sin(q),0],[0,0,0,0]]
def Rz(q): return [[cos(q),-sin(q),0,0],[sin(q),cos(q),0,0],[0,0,1,0],[0,0,0,1]]
def dRz(q):return [[-sin(q),-cos(q),0,0],[cos(q),-sin(q),0,0],[0,0,0,0],[0,0,0,0]]
def Tx(q): return [[1,0,0,q],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
def dTx(): return [[0,0,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
def Ty(q): return [[1,0,0,0],[0,1,0,q],[0,0,1,0],[0,0,0,1]]
def dTy(): return [[0,0,0,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]]
def Tz(q): return [[1,0,0,0],[0,1,0,0],[0,0,1,q],[0,0,0,1]]
def dTz(): return [[0,0,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]]


def fk(q,theta): return dot([Tz(l1),Rz(q[0]),Rz(theta[0]),Tz(q[1]),Tz(theta[1]),Ty(l3),Ty(q[2]),Ty(theta[2])])
def ik(x,y,z): return [arctan2(y, x), z-l1, (x**2+y**2)**.5 - l3]
def Jth(q, theta):
    H = fk(q, theta)
    H[0:3, 3] = 0
    inv_H = transpose(H)
    dH = dot([Tz(l1),Rz(q[0]),dRz(theta[0]),Tz(q[1]),Tz(theta[1]),Ty(l3),Ty(q[2]),Ty(theta[2]),inv_H])
    J1 = v([dH[0, 3], dH[1, 3], dH[2, 3], dH[2, 1], dH[0, 2], dH[1, 0]])
    dH = dot([Tz(l1),Rz(q[0]),Rz(theta[0]),Tz(q[1]),dTz(),Ty(l3),Ty(q[2]),Ty(theta[2]),inv_H])
    J2 = v([dH[0, 3], dH[1, 3], dH[2, 3], dH[2, 1], dH[0, 2], dH[1, 0]])
    dH = dot([Tz(l1),Rz(q[0]),Rz(theta[0]),Tz(q[1]),Tz(theta[1]),Ty(l3),Ty(q[2]),dTy(),inv_H])
    J3 = v([dH[0, 3], dH[1, 3], dH[2, 3], dH[2, 1], dH[0, 2], dH[1, 0]])
    return h([J1, J2, J3])

experiments = 30
A1, A2 = zeros((3, 3)),zeros(3)

for i in range(experiments):
    q = h([uniform(-pi,pi), uniform(0,1,2)])
    W = uniform(-1000, 1000, 6)

    jTheta = Jth(q, theta)
    dt = dot([jTheta, inv(kTheta), transpose(jTheta), W]) + normal(loc=0.0,scale=1e-5)
    jTheta = jTheta[0:3, :]
    dt = dt[0:3]
    W = W[0:3]

    A = zeros(jTheta.shape)
    for i in range(jTheta.shape[1]):
        j = jTheta[:,i]
        A[:, i] = outer(j, j).dot(W)
    A1 += transpose(A).dot(A)
    A2 += transpose(A).dot(dt)

kTheta2 = diag(divide(1, inv(A1).dot(A2)))

W = [-440, -1370, -1635, 0, 0, 0]
r,H = 0.1,0.1
points = 50
alpha = linspace(0, 2 * pi, points)
tDes = stack([r * cos(alpha),r * sin(alpha), H * ones(points)])
tUnCal,tCal,tUpd = zeros(tDes.shape),zeros(tDes.shape),zeros(tDes.shape)

for i in range(points):
    jTheta = Jth(ik(tDes[0,i], tDes[1,i], tDes[2,i]), theta)
    dt = dot([jTheta, inv(kTheta), transpose(jTheta), W]) + normal(loc=0.0,scale=1e-5)
    tUnCal[:, i] = tDes[:, i] + dt[0:3]
    tUpd[:,i] = tDes[:, i] - dt[0:3]
    jTheta = Jth(ik(tUpd[0,i], tUpd[1,i], tUpd[2,i]), theta)
    dt = dot([jTheta, inv(kTheta2), transpose(jTheta), W]) + normal(loc=0.0,scale=1e-5)
    tCal[:, i] = tUpd[:, i] + dt[0:3]

fig = plt.figure(figsize=(9, 6))
ax = plt.axes(projection='3d')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.plot3D(tDes[0], tDes[1], tDes[2], c='green',label='Desired trajectory')
ax.plot3D(tUnCal[0], tUnCal[1], tUnCal[2], c='blue',label='Uncalibrated trajectory')
ax.plot3D(tCal[0], tCal[1], tCal[2], c='red',label='Calibrated trajectory')

plt.legend()
plt.show()
