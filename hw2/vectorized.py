import numpy as np
A = np.array([[-2,-3],[1,0]])
B = np.array([[1,1],[1,0]])
x = np.array([[-1],[1]])

C = np.linalg.inv(A)

print(A@C)
print(C@A)
print((C@A).shape)

print(A@x)
print(A.T@A)
print((A@B).T)
print(A@x - B@x)
print(np.linalg.norm(x))
print(np.linalg.norm(A@x - B@x))

print(A[:,0])
D = B[:]
D[:,0] = x[:,0]
print(D)

print(A[:,0]*A[:,1])
