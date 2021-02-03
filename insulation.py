# Task
# T_summer vs tauClo
# T_winter vs rhoClo
# when Tenv change, can tauClo or rhoClo also change
# from 0 to 1 to meet the physical mechanisms?

# constants
# degree C
TBody = 34
TSummer = [x for x in range(20,27)]
TWinter = [x for x in range(12,20)]
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
# kAir = 0.024
# # m
# thickAir = 1**(-3)
#
# #For winter
# # rhoClo = 0.9
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
import fqs
import numpy as np
p = np.array([[1, 4, 6, 4, 1]])
roots = fqs.quartic_roots(p)
print(roots)


