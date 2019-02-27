# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 16:10:03 2019

@author: helen
"""

import numpy as np
import numpy.linalg as la
from numpy import inf
from numpy import pi
from numpy import sin, cos
import matplotlib.pyplot as plt

#For inspirasjon:

def A_laplace(M):
    # Construct the discrete laplacian, that is
    # A = blocktridiag(I,T,I) is a M^2xM^2 matrix where
    # T = tridiag(1,-4,1) is a MxM matrix
    M2 = M**2
    A = -4*np.eye(M2)            # The diagonal matrix
    for i in range(M2-1):
        A[i,i+1] = 1
        A[i+1,i] = 1
    for i in range(M-1,M2-1,M):
        A[i,i+1] = 0
        A[i+1,i] = 0
    for i in range(M2-M):      # The block sub- and sup diagonal
        A[i,i+M] = 1
        A[i+M,i] = 1
    return A

def tridiag(c, a, b, N):
    # Returns a tridiagonal matrix A=tridiag(c, a, b) of dimension N x N.
    e = np.ones(N)        # array [1,1,...,1] of length N
    A = c*np.diag(e[1:],-1)+a*np.diag(e)+b*np.diag(e[1:],1)
    return A

def A(h):
    M = 1/h
    
    N = 4 * M**2 - 1 
    A = -4*np.eye(N)
    
    rows = int((2/h) - 1)
    full_cols = int((1/h) + 2)
    
    indeks = 0
    
    for j in range(rows):
        #Nå itererer du over 0-indekserte rader
        
        #Hvis første linje:
        if (j == 0):
            #Fyll inn for nedre boundary
            for l in range(full_cols):
                d = full_cols + j
                if(l == 0):
                    A[indeks, indeks + 1] = 2
                    A[indeks, indeks + d] = 1
                    
                elif (l == full_cols - 1):
                    A[indeks, indeks - 1] = 2
                    A[indeks, indeks + d] = 2
                    
                else:
                    A[indeks, indeks - 1] = 1
                    A[indeks, indeks + 1] = 1
                    A[indeks, indeks + d] = 1
                indeks += 1
                        
        elif (j == (rows - 1)):
            #Fyll inn for øvre boundary
            d = full_cols + j
            
            for m in range(full_cols + j):
                #Fyll på
                print("Hei, nå er vi ved øvre boundary, j er: ", j)
                if(m == 0):
                    A[indeks, indeks + 1] = 2
                    A[indeks, indeks - (d-1)] = 1
                elif(m == full_cols + j -1):
                    A[indeks, indeks -1] = 2
                else:
                    A[indeks,indeks - 1] = 1
                    A[indeks,indeks + 1] = 1
                    #A[indeks,indeks + d + 3] = 1
                    A[indeks,indeks - d + 1] = 1
                
                indeks += 1
        else: 
                    
            for k in range(full_cols + j):
                # nå itererer du over elementer i raden, 0-indeksert
                #lag if-setninger som sier noe om blokk og linje:
                #nummeret er nå j + k
                
                print("Hello, inne i inner løkke, nå er j:", j)
                
                d = full_cols + j  
                print(indeks)
                
                if (k == 0):
                    #fyll inn for vertikal
                    print("k == 0")
                    A[indeks,indeks + 1] = 2
                    A[indeks,indeks + d + 1] = 1
                    A[indeks,indeks - d] = 1                
                    
                if (k == full_cols + j - 1):
                    #fyll inn for diagonal
                    print("k == full_cols + j - 1")
                    A[indeks,indeks - 1] = 2
                    A[indeks,indeks + d + 1] = 2
                else :
                    #fyll inn for inner lines
                    A[indeks,indeks - 1] = 1
                    A[indeks,indeks + 1] = 1
                    A[indeks,indeks + d + 1] = 1
                    A[indeks,indeks - d] = 1
                
                indeks += 1
    b = np.zeros(int(N))
    b[int(N-1)] = -2
    b[int((N-1)-(full_cols + rows -2)):int((N-1))] = -1
    return A, b, h

def solve_U(A,b):
    U = la.solve(A,b)
    return U 

def sett_sammen_U(U,h):
    #U skal være på trekantformen
    #anta at du vet h
    
    
    
    
