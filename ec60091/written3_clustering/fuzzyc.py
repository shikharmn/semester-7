import numpy as np

c1, c2 = None, None
variables = np.array([[1, 6], [2, 5], [3, 8], [4, 4], [5, 7], [6,9]])
matrix = np.array([ [0.5, 0.2, 0.1, 0.3, 0.2, 0.9], [0.2, 0.9, 0.5, 0.5, 0.8, 0.2] ])
# matrix = np.array([ [0.8, 0.9, 0.7, 0.3, 0.5, 0.2], [0.3, 0.8, 0.5, 0.9, 0.8, 0.1] ])
# matrix[1] = 1-matrix[0]
# matrix = np.random.random((2,6))
for i in range(4):
    c1, c2 = np.array([0.0, 0]), np.array([0.0, 0])
    for j in range(6):
        c1+=variables[j]*(matrix[0][j]**2)
        c2+=variables[j]*(matrix[1][j]**2)
    print(c1, c2)
    c1/=np.sum(matrix[0]**2)
    c2/=np.sum(matrix[1]**2)
    print("=========Centres======")
    print(c1, c2)

    for j in range(6):
        num, denom = np.linalg.norm(c1-variables[j])**-1, np.linalg.norm(c1-variables[j])**-1 + np.linalg.norm(c2-variables[j])**-1
        matrix[0][j] = num/denom
        # print(num, denom)
        # print(num**-1)
        num, denom = np.linalg.norm(c2-variables[j])**-1, np.linalg.norm(c1-variables[j])**-1 + np.linalg.norm(c2-variables[j])**-1
        matrix[1][j] = num/denom
            
    print("=========Matrix======")
    print(matrix)
    print("=======================")
    print()