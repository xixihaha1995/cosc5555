import numpy as np
import fqs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
import scipy.interpolate

import matplotlib
# fig = plt.figure()
# ax = plt.axes(projection='3d')

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
# opereation = "heating"
opereation = "indoor cooling"
# opereation = "outdoor cooling"
#
if (opereation == "heating"):
    TEnvLow = 12 + 273
    TEnvHigh = 20 + 273
elif(opereation == "outdoor cooling"):
    TEnvLow = 35 + 273
    TEnvHigh = 40 + 273
else:
    TEnvLow = 25 + 273
    TEnvHigh = 28 + 273
# watts/K/m2
Metab = 58

# sigma unit W*(m^-2)*(K^-4)
sigma = 5.67*10**(-8)
epsilonSkin = 1
epsilonEnv = 1
FskToCl = 1
FclToSk = 1

# # the following has not been implemented
# AreaBody = 2
# AreaEnv = 10
# FEnvToCl = 0.28
# FclToEnv = 0.72

# enclosed Air
# W*(m^-2)*(K^-1)
kAir = 0.024
kClo = 0.047
# m
thickAir = 2*10**(-3)
thickClo = 0.5*10**(-3)

qRS = sigma * TBody ** 4
# qRE = sigma * TEnv ** 4

X,Y,Z = [],[],[]
Epsilon = []

for TEnv in range(TEnvLow,TEnvHigh):
    tauCloList = []
    rhoCloList = []
    hConvCEList = []
    epsilonCloList = []
    qRE = sigma * TEnv ** 4

    for tauClo in np.linspace(0, 1, 101):
        # tauClo = 0.03
        for rhoClo in np.linspace(0, 1, 101):
            # rhoClo = 0.3
            if (tauClo + rhoClo) > 1:
                continue
            epsilonClo = 1 - tauClo - rhoClo
            if(epsilonClo<0 or epsilonClo > 1):
                continue
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
            # print("TCloOne:" + str(TCloOneRoots[0][1].real)+", TEnv:" + str(TEnv))
            if (TCloOneRoots[0][1].real > 0 and TCloOneRoots[0][1].real < TEnv):
                TCloOne = TCloOneRoots[0][1].real
            else:
                continue
            # TCloTwo = TCloOne - Metab/kClo*thickClo
            TCloTwo = TCloOne
            # print("TCloOne:"+str(TCloOne)+", TEnv:"+ str(TEnv))
            qRCloTwo = epsilonClo *sigma*TCloTwo**4
            qConvCE = Metab - tauClo*qRS+(epsilonClo-rhoClo)*qRE - qRCloTwo
            hConvCE = qConvCE/(TCloTwo - TEnv)
            if(hConvCE <= 0 or hConvCE > 50  ):
                continue
            tauCloList.append(tauClo)
            rhoCloList.append(rhoClo)
            hConvCEList.append(hConvCE)
            epsilonCloList.append(epsilonClo)

    xtau = np.array(tauCloList)
    yrho = np.array(rhoCloList)
    zhconv = np.array(hConvCEList)
    epcl = np.array(epsilonCloList)
    X.append(xtau)
    Y.append(yrho)
    Z.append(zhconv)
    Epsilon.append(epcl)
plots = zip(X,Y,Z)
# for idx,itm in enumerate(X):
#     print(len(itm))
# print(X)
# print(Y)
# print(Epsilon)
# print(Z)


def loop_plot(plots,len):
    figs = plt.figure(figsize=(30, 30))
    xyLabel = True
    for idx, plot in enumerate(plots):
        ax=figs.add_subplot(1,len,idx+1, projection='3d')
        try:
            # ax.plot_trisurf(plot[0],plot[1],plot[2])
            ax.scatter(plot[0], plot[1], plot[2],marker="o")
        except RuntimeError:
            continue
        ax.set_title("TEnv: "+str(TEnvLow+1+idx)+"K")
        if(xyLabel):
            ax.set_xlabel('tau')
            ax.set_ylabel('rho')
            ax.set_zlabel('hConv')
            # plt.xlim([0, 1])
            # plt.ylim([0, 1])
            # xyLabel =False
    return figs
figs = loop_plot(plots,len(X))


figs.suptitle("Smart Clothing "+opereation+" Maps",fontsize=36)
figs.savefig(opereation+" from "+str(TEnvLow) +"K to "+str(TEnvHigh)+"K.png")
plt.show()









