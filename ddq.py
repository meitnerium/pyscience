import numpy as np
import matplotlib.pyplot as plt

class Grid():
    def __init__(self,value, ndim=1, dim=[1024],name=["X"],unit=["u.a."]):
        self.ndim = ndim
        self.dim = dim
        self.value = value
        self.name = name
        self.unit = unit

class Potential():
    def __init__(self,ncanal,grid,value,name=["Energy"],unit=["u.a."]):
        self.grid = grid
        self.value = value
        self.ncanal = ncanal
        self.name = name
        self.unit = unit


def morse_nu(xmu,smalla,diss,nu,alpha,r,requ,e0):
    biga = (np.sqrt(2.0*xmu))/smalla
    bigc = biga*np.sqrt(diss)
    enu = -((bigc-nu-.5)**2.0)/biga**2.0
    alpha = bigc-nu-0.50
    arg=np.exp(-smalla*(r-requ))
    x=2.0*bigc*arg
    m = 2*(int(bigc)-nu)-1
    morse = np.polynomial.laguerre.Laguerre() #coef, domain=None, window=None)
    morse = morse * (x**(int(bigc)))
    morse = morse / (x**nu)
    morse = morse / np.sqrt(x)
    morse = morse * np.exp(-x/2.0)

    norm = smalla * np.math.factorial(nu) * (2.0*bigc - 2.0*nu - 1.0)
    norm = norm / np.math.factorial(m+nu)

    norm = np.sqrt(norm)
    morse = morse*norm
    return morse


def morse(z0,s,r,req,x1):
    return z0*(np.exp(-2.0*s*(r-req)) - 2.0*x1*np.exp(-s*(r-req)))

def xmu12(s,y,r,req):
    #
    return 1.070 + (.3960/(s*y))*(1.0-np.exp(-s*y*(r-req)))
    #if((r>12.0) and (xmu12 > 0.5*r)):
    #    xmu12 = 0.50*r


def eval(p0,rc0,alpha, r):
#subroutine eval(cw1, cw2, delr, rdeb, p0, rc0, alpha, npos)
    cw1 = np.zeros(len(r), dtype=np.complex64)
    cw2 = np.zeros(len(r), dtype=np.complex64)
    cpoi = np.sqrt(np.sqrt(2.0 * alpha / np.pi + 1j*(0.0)))
    arg = (-alpha * (r - rc0) ** 2 + 1j * (p0 * (r - rc0)))
    cw1 = cpoi * np.exp(arg)

    return cw1,cw2


def zexptdt(r,masse,dt):
    xk = np.zeros(len(r))
    etdt = np.zeros(len(r), dtype=np.complex64)
    xk1 = 2.0 * np.pi / (len(r) * (r[1]-r[0]))
    for nr in range(len(r)):
        if nr<len(r/2):
            xk[nr] = (nr) * xk1
        else:
            xk[nr] = -(len(r) - nr) * xk1
        arg = ((xk[nr] * xk[nr]) / (2.0 * masse)) * dt
        etdt[nr] = np.exp(-1j * arg)
    return etdt


time = Grid(ndim=1,dim=1024*48,value=np.linspace(0,8000,1024*48))
x = Grid(ndim=1,dim=1024,value=np.linspace(0.1,15.0,1024))
#print(x.value)
data = np.array([[np.array(morse(z0=.1026277, s=.72, r=x.value, req=2., x1=1.)),
                  np.array(xmu12(s=.72,y = -.055,r=x.value,req=2.))],
                 [np.array(xmu12(s=.72,y = -.055,r=x.value,req=2.)),
                  np.array(morse(z0=.1026277, s=.72, r=x.value, req=2., x1=-1.11))]])
potH2 = Potential(grid=x, ncanal=2,
                  value=data)
plt.plot(potH2.grid.value,potH2.value[0,0],potH2.grid.value,potH2.value[1,1])
plt.xlabel(potH2.grid.name[0]+" ("+potH2.grid.unit[0]+")")
plt.ylabel(potH2.name[0]+" ("+potH2.unit[0]+")")
plt.show()
plt.close()

