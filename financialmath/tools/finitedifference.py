from dataclasses import dataclass
import numpy as np
from scipy import sparse

@dataclass
class OneFactorImplicitScheme:

    up_matrix : np.array
    mid_matrix: np.array 
    down_matrix: np.array
    M : int 
    N : int

    def compute_matrix(self,down:np.array, mid: np.array, 
                                 up:np.array) -> sparse.csc_matrix: 
        tmat = sparse.diags(diagonals =[down, mid, up],
                            offsets = [-1, 0, 1],
                            shape=(self.M, self.M))
        tmat = tmat.todense()
        i = 0
        tmat[i,:] = 2*tmat[i+1,:] - tmat[i+2,:]
        i = self.M-1
        tmat[i,:] = 2*tmat[i-1,:] - tmat[i-2,:]
        return sparse.csc_matrix(tmat)
    
    def transition_matrixes(self) -> list[sparse.csc_matrix]: 
        upm, midm, downm = self.up_matrix, self.mid_matrix, self.down_matrix
        output = []
        for i in range(1,self.N):
            vup, vdown, vmid =  upm[:,i], downm[:,i], midm[:,i]
            tmat = self.compute_matrix(vdown,vmid,vup)
            output.append(tmat)
        return output

