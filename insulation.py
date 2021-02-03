import numpy as np
import fqs

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
TEnv = 26 + 273
Metab = 58
TSummer = [x for x in range(20, 27)]
TWinter = [x for x in range(12, 20)]
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

qRE = sigma*TEnv**4
qRS = sigma*TBody**4
for tauClo in np.linspace(0, 1, 11):
    for rhoClo in np.linspace(0, 1, 11):
        if (tauClo + rhoClo) > 1:
            continue
        epsilonClo = 1 - tauClo - rhoClo
        A = -sigma*epsilonClo
        B = 0
        C = 0
        D = -1*kAir/thickAir
        E = kAir/thickAir*(TBody)-tauClo*qRE + (epsilonClo - rhoClo) * qRS - Metab
        p = np.array([[A, B, C, D, E]])
        TCloOneRoots = fqs.quartic_roots(p)
        if (TCloOneRoots[0][1].real > 0):
            TCloOne = TCloOneRoots[0][1].real
        else:
            continue
        TCloTwo = TCloOne - Metab/kClo*thickClo
        print("TCloOne:"+str(TCloOne)+", TCloTwo:"+ str(TCloTwo))







#For winter
# rhoClo = 0.9
#
#
# for TEnv in TSummer:
#     tauClo = 0.03
#     rhoClo = 1 - epsilonClo - tauClo
#     qRadSkin = FskToCl*sigma*(TBody+273)**4
#     qRadEnv = FEnvToCl*sigma*(TEnv+273)**4
#     qTemp = epsilonClo*qRadSkin - rhoClo*qRadSkin - tauClo * qRadEnv
#
#     print(qTemp)



