from dataclasses import dataclass
import numpy as np
from scipy import sparse
from abc import ABC, abstractmethod
from enum import Enum


class SchemeAbstract(ABC): 

    @abstractmethod
    def transition_matrixes(self) -> list[sparse.csc_matrix]: 
        pass

@dataclass
class ImplicitScheme(SchemeAbstract): 

    volmatrix : np.array
    dx : float 
    dt : float 
    r : float 
    q : float 
    N : int

    def discount_factor(self) -> float: 
        return 1/(1+self.r*self.dt)

    def up_move(self) -> np.array: 
        p = self.dt*(self.volmatrix**2)/(2*(self.dx**2))
        u = (self.r-self.q-(self.volmatrix**2)/2)*self.dt/(2*self.dx)
        return p + u

    def down_move(self)-> np.array: 
        p = self.dt*(self.volmatrix**2)/(2*(self.dx**2))
        d = (self.r-self.q-(self.volmatrix**2)/2)*self.dt/(2*self.dx)
        return p - d

    def mid_move(self)-> np.array: 
        p = self.dt*(self.volmatrix**2)/(2*(self.dx**2))
        return 1-2*p

    def create_transition_matrix(self,vectors: dict) -> sparse.csc_matrix: 
        M = self.volmatrix.shape[0]
        tmat = sparse.diags(diagonals =[vectors['down'], vectors['mid'], vectors['up']],
                            offsets = [-1, 0, 1],
                            shape=(M, M))
        tmat = tmat.todense()
        i = 0
        tmat[i,:] = 2*tmat[i+1,:] - tmat[i+2,:]
        i = M-1
        tmat[i,:] = 2*tmat[i-1,:] - tmat[i-2,:]

        return sparse.csc_matrix(tmat)

    def transition_matrixes(self) -> list[sparse.csc_matrix]: 
        df = self.discount_factor()
        up = df*self.up_move()
        down = df*self.down_move()
        mid = df*self.mid_move()
        vectors = [{'up': up[:,i],'mid': mid[:,i], 'down': down[:,i]}
                    for i in range(0, self.N)]
        return [self.create_transition_matrix(vectors = v) for v in vectors]

class OneFactorSchemeList(Enum): 
    implicit = 'Implicit Scheme' 
    crancknicolson = 'Crank-Nicolson Scheme'

class OneFactorScheme(Enum): 
    implicit = ImplicitScheme 
    crancknicolson = None 

    def get_scheme(scheme: OneFactorSchemeList) -> classmethod:
        try: return [s.value for s in list(OneFactorScheme) if s.name == scheme.name][0]
        except IndexError: return None 


