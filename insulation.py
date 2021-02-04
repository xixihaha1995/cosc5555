import numpy as np
import fqs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
import scipy.interpolate

import matplotlib
fig = plt.figure()
ax = plt.axes(projection='3d')

# # quartic solver

#
# p = np.array([[1, 4, 6, 4, 1]])
# roots = fqs.quartic_roots(p)
# print(roots[0][0].real)
# print(type(roots[0][0].real))

# Task
# T_summer vs tauClo
# T_winter vs rhoClo
# when Tenv change, can tauClo or rhoClo also change
# from 0 to 1 to meet the physical mechanisms?

# constants
# degree C
TBody = 34 + 273

TEnvLow = 20 + 273
TEnvHigh = 26 + 273
Metab = 58
# watts/K/m2
qBar = 58.2
# sigma unit W*(m^-2)*(K^-4)
sigma = 5.67*10**(-8)
epsilonSkin = 1
epsilonEnv = 1
FskToCl = 1
FclToSk = 1

AreaBody = 2
AreaEnv = 10
FEnvToCl = 0.28
FclToEnv = 0.72

# enclosed Air
# W*(m^-2)*(K^-1)
kAir = 0.024
kClo = 0.047
# m
thickAir = 1*10**(-3)
thickClo = 0.5*10**(-3)

qRS = sigma * TBody ** 4

X,Y,Z = [],[],[]

for TEnv in range(TEnvLow,TEnvHigh):
    tauCloList = []
    rhoCloList = []
    hConvCEList = []
    qRE = sigma * TEnv ** 4

    for tauClo in np.linspace(0, 1, 101):
        for rhoClo in np.linspace(0, 1, 101):
            if (tauClo + rhoClo) > 1:
                continue
            epsilonClo = 1 - tauClo - rhoClo
            A = -sigma*epsilonClo
            B = 0
            C = 0
            D = -1*kAir/thickAir
            E = kAir/thickAir*(TBody)-tauClo*qRE + (epsilonClo - rhoClo) * qRS - Metab
            p = np.array([[A, B, C, D, E]])
            try:
                TCloOneRoots = fqs.quartic_roots(p)
            except ZeroDivisionError:
                continue
            if (TCloOneRoots[0][1].real > 0):
                TCloOne = TCloOneRoots[0][1].real
            else:
                continue
            TCloTwo = TCloOne - Metab/kClo*thickClo
            # print("TCloOne:"+str(TCloOne)+", TCloTwo:"+ str(TCloTwo))
            qRCloTwo = sigma*TCloTwo**4
            qConvCE = Metab - tauClo*qRS+(epsilonClo-rhoClo)*qRE - qRCloTwo
            hConvCE = qConvCE/(TCloTwo - TEnv)
            if(hConvCE <= 0 or hConvCE > 50 ):
                continue
            tauCloList.append(tauClo)
            rhoCloList.append(rhoClo)
            hConvCEList.append(hConvCE)
    xtau = np.array(tauCloList)
    yrho = np.array(rhoCloList)
    zhconv = np.array(hConvCEList)
    X.append(xtau)
    Y.append(yrho)
    Z.append(zhconv)
    # print(xtau.shape)
    # ax.plot_trisurf(xtau,yrho,zhconv)
    # ax.set_xlabel('tau')
    # ax.set_ylabel('rho')
    # ax.set_zlabel('hConv')
    # ax.set_title("Max TEnv: "+str(TEnv)+"K")
plots = zip(X,Y,Z)
def loop_plot(plots,len):
    figs = plt.figure()
    for idx, plot in enumerate(plots):
        ax=figs.add_subplot(1,len,idx+1, projection='3d')
        ax.plot_trisurf(plot[0],plot[1],plot[2])
        ax.set_title("Max TEnv: "+str(273+16+1+idx)+"K")
        if(idx == 0):
            ax.set_xlabel('tau')
            ax.set_ylabel('rho')
            ax.set_zlabel('hConv')
    return figs
figs = loop_plot(plots,len(X))
figs.savefig("TEnv from "+str(TEnvLow) +"K to "+str(TEnvHigh)+"K.png")
plt.show()









