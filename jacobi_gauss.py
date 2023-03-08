import numpy as np
import time as t

y=10
iterace=1000000
var=1

"""
a) y=10
b) y=2
c) y=4/5
interace = cislo max. poctu iteraci
var = 1 -> Jacobi method
var = 2 -> gauss seidel

vsechna zadani pro matice 20x20
"""

startTime=t.time()

#init A
A=np.zeros((20,20))                                               
A.flat[0::21]=y
A.flat[1::21]=-1
A.flat[20::21]=-1

#init b
b=np.zeros((20,1))
b[0]=y-1
b[1:19]=y
b[19]=y-1


#init x
x=np.zeros((20,20),dtype=np.double)

def jacobi(A,b,iterace):
    tolerance=1e-6
    D=np.diagonal(A)
    T=A-np.diag(D) #vektor diag. elem
    
    for k in range(iterace):
        xprev=x.copy()
        x[:]=(b-np.dot(T,x)) / D
        
        #converg
        if np.linalg.norm(x-xprev,ord=np.inf) / np.linalg.norm(x,ord=np.inf)<tolerance:
            break
    return x,k


def gs(A,b,iterace):
    tolerance=1e-6
    for k in range(iterace):
        xprev=x.copy()
        for i in range(A.shape[0]): #check rows
            x[i]=(b[i]-np.dot(A[i,:i], x[:i])-np.dot(A[i,(i+1):], xprev[(i+1):])) / A[i,i]
            
        #converg
        if np.linalg.norm(x-xprev,ord=np.inf) / np.linalg.norm(x,ord=np.inf)<tolerance:
            break
            
    return x,k


if var==1:
    print(jacobi(A,b,iterace))
    print("\n ---%s s runtime ---" %(t.time()-startTime))
elif var==2:
    print(gs(A,b,iterace))
    print("\n ---%s s runtime ---" %(t.time()-startTime))




'''
kontrola diagonalni dominantnosti matice A
https://wikijii.com/wiki/Diagonally_dominant_matrix
'''
def dom() :
    for i in range(20): #prochazeni radku    
        s=0
        for k in range(20): #soucet radku
            s=s+abs(A[i][k])    
        s=s-abs(A[i][i]) #odstraneni diag
 
        #diag el < nondiag
        if (abs(A[i][i])<s) :
            return print("d.nedominantni")
    return print("d.dominantni")

dom()

#kontrola definitnosti
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)
#print(is_pos_def(A))