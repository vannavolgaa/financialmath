from financialmath.tools.probability import ProbabilityDistribution
import numpy as np
import matplotlib.pyplot as plt

N=1000
M=2
corr_matrix = np.reshape([1,0.5,0.5,1],(M,M))
mat = ProbabilityDistribution.skewed_student_t({'df':15, 'tau':4}).correlated_random_matrix(corr_matrix=corr_matrix, N=N, M=M)

plt.plot(np.cumsum(mat[0,:]))
plt.plot(np.cumsum(mat[1,:]))
plt.show()