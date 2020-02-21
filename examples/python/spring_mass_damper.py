import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

from pspace import PSPACE
from tacs import TACS, elements

class SMDUpdate:
    def __init__(self, elem):
        self.element = elem
        return

    def update(self, vals):
        self.element.setMass(vals[0])
        self.element.setDamping(vals[1])
        self.element.setStiffness(vals[2])
        #self.element.m = vals[0]
        #self.element.c = vals[1] 
        #self.element.k = vals[2]
        return

# Define an element in TACS using the pyElement feature
class SpringMassDamper(elements.pyElement):
    def __init__(self, num_disps, num_nodes, m, c, k):
        self.m = m
        self.c = c
        self.k = k    

    def getInitConditions(self, index, X, v, dv, ddv):
        '''Define the initial conditions'''
        v[0] = -0.5
        dv[0] = 1.0
        return

    def addResidual(self, index, time, X, v, dv, ddv, res):
        '''Add the residual of the governing equations'''
        res[0] += self.m*ddv[0] + self.c*dv[0] + self.k*v[0]
        return    

    def addJacobian(self, index, time, alpha, beta, gamma, X, v, dv, ddv, res, mat):
        '''Add the Jacobian of the governing equations'''
        res[0] += self.m*ddv[0] + self.c*dv[0] + self.k*v[0]
        mat[0] += alpha*self.k + beta*self.c + gamma*self.m
        return

def createAssembler(m=5.0, c=0.5, k=5.0, pc=None):
    num_disps = 1
    num_nodes = 1
    #spr = SpringMassDamper(num_disps, num_nodes, m, c, k)
    spr = PSPACE.PySMD(m, c, k)
    elem = spr
    ndof_per_node = 1
    num_owned_nodes = 1
    num_elems = 1
    if pc is not None:
        cb = SMDUpdate(spr)
        elem = PSPACE.PyStochasticElement(spr, pc, cb)
        ndof_per_node = ndof_per_node*pc.getNumBasisTerms()
    
    # Add user-defined element to TACS
    comm = MPI.COMM_WORLD
    assembler = TACS.Assembler.create(comm, ndof_per_node, num_owned_nodes, num_elems)

    ptr = np.array([0, 1], dtype=np.intc)
    conn = np.array([0], dtype=np.intc)
    assembler.setElementConnectivity(ptr, conn)
    assembler.setElements([elem])
    assembler.initialize()

    return assembler

def moments(bdf, num_steps, nterms):
    umean = np.zeros((num_steps+1))
    udotmean = np.zeros((num_steps+1))
    uddotmean = np.zeros((num_steps+1))
    time = np.zeros((num_steps+1))

    uvar = np.full_like(umean, 0)
    udotvar = np.full_like(udotmean, 0)
    uddotvar = np.full_like(uddotmean, 0)

    # Compute mean and variance at each time step
    for k in range(0, num_steps+1):
        # Extract the state vector
        t, uvec, udotvec, uddotvec = bdf.getStates(k)
        u = uvec.getArray()
        udot = udotvec.getArray()
        uddot = uddotvec.getArray()
        
        # Compute moments
        time[k] = t
        umean[k] = u[0]
        udotmean[k] = udot[0]
        uddotmean[k] = uddot[0]
        for i in range(1,nterms):
            uvar[k] += u[i]**2 
            udotvar[k] += udot[i]**2
            uddotvar[k] += uddot[i]**2
        
    return time, umean, udotmean, uddotmean, uvar, udotvar, uddotvar

pfactory = PSPACE.PyParameterFactory()
y1 = pfactory.createExponentialParameter(mu=4.0, beta=1.0, dmax=3)
y2 = pfactory.createUniformParameter(a=0.25, b=0.75, dmax=3)
y3 = pfactory.createNormalParameter(mu=5.0, sigma=0.5, dmax=3)

basis_type = 1
pc = PSPACE.PyParameterContainer(basis_type)
pc.addParameter(y1)
pc.addParameter(y2)
pc.addParameter(y3)

pc.initialize()

# Create TACS
m = 1.0
c = 0.5
k = 5.0
tf = 100.0
assembler = createAssembler(m=m, c=c, k=k, pc=pc)

# Create Integrator
t0 = 0.0
tf = 10.0
num_steps = 100
order = 2
integrator = TACS.BDFIntegrator(assembler, t0, tf, num_steps, order)
integrator.setPrintLevel(1)
integrator.integrate()

nterms = pc.getNumBasisTerms()
time, umean, udotmean, uddotmean, uvar, udotvar, uddotvar = moments(integrator, num_steps, nterms)

## def getJacobian(pc):
## M = pc.getNumQuadraturePoints()
## N = pc.getNumBasisTerms()
## A = np.zeros((N, N))
## for q in range(M):
##     wq, zq, yq = pc.quadrature(q)
##     for i in range(N):
##         psiziq = pc.basis(i, zq)
##         for j in range(N):
##             psizjq = pc.basis(j, zq)                
##             A[i,j] += wq*psiziq*psizjq*(yq[0]+yq[1]+yq[2])
## return A

## from pspace.plotter import plot_jacobian
## A = getJacobian(pc)
## plot_jacobian(A, 'sparsity.pdf')

# Compute moments

###################################################################
# plot results
###################################################################

# Configure 
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'

# Optionally set font to Computer Modern to avoid common missing
# font errors
params = {
  'axes.labelsize': 20,
  'legend.fontsize': 14,
  'xtick.labelsize': 20,
  'ytick.labelsize': 20,
  'text.usetex': True}
plt.rcParams.update(params)

# Latex math
plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath}']
#plt.rcParams['font.family'] = 'sans-serif'
#plt.rcParams['font.sans-serif'] = 'courier'
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['lines.color'] = 'r'

# Make sure everything is within the frame
plt.rcParams.update({'figure.autolayout': True})

# Set marker size
markerSize = 7.0 #11.0
mew = 2.0

# bar chart settings
lalpha    = 0.9
rev_alpha = 0.9/1.5

# These are the "Tableau 20" colors as RGB.    
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    
  
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)

plt.figure()
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.plot(time, umean    , '-', label='$u(t)$'       , mew=mew, ms=markerSize, color=tableau20[2], mec='black')
plt.plot(time, udotmean , '-', label='$\dot{u}(t)$' , mew=mew, ms=markerSize, color=tableau20[4], mec='black')
plt.plot(time, uddotmean, '-', label='$\ddot{u}(t)$', mew=mew, ms=markerSize, color=tableau20[6], mec='black')
plt.xlabel('time [s]')
plt.ylabel('expectation')
plt.legend(loc='best', frameon=False)
plt.savefig('smd-galerkin-expectation.pdf',
            bbox_inches='tight', pad_inches=0.05)

plt.figure()
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.plot(time, uvar    , '-', label='$u(t)$'       , mew=mew, ms=markerSize, color=tableau20[2], mec='black')
plt.plot(time, udotvar , '-', label='$\dot{u}(t)$' , mew=mew, ms=markerSize, color=tableau20[4], mec='black')
plt.plot(time, uddotvar, '-', label='$\ddot{u}(t)$', mew=mew, ms=markerSize, color=tableau20[6], mec='black')
plt.xlabel('time [s]')
plt.ylabel('variance')
plt.legend(loc='best', frameon=False)
plt.savefig('smd-galerkin-variance.pdf',
            bbox_inches='tight', pad_inches=0.05)


plt.figure()
sigma = 1.0
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.plot(time, umean    , '-', label='${E}[{u}(t)]\pm{S}[{u}(t)]$', mew=mew, ms=markerSize, color=tableau20[2], mec='black')
plt.fill_between(time, umean, umean + sigma*np.sqrt(uvar), color=tableau20[3], alpha=0.5)
plt.fill_between(time, umean, umean - sigma*np.sqrt(uvar), color=tableau20[3], alpha=0.5)

plt.plot(time, udotmean , '-', label='${E}[\dot{u}(t)]\pm{S}[\dot{u}(t)]$', mew=mew, ms=markerSize, color=tableau20[4], mec='black')
plt.fill_between(time, udotmean, udotmean + sigma*np.sqrt(udotvar), color=tableau20[5], alpha=0.5)
plt.fill_between(time, udotmean, udotmean - sigma*np.sqrt(udotvar), color=tableau20[5], alpha=0.5)

plt.plot(time[1:], uddotmean[1:], '-', label='${E}[\ddot{u}(t)]\pm{S}[\ddot{u}(t)]$', mew=mew, ms=markerSize, color=tableau20[6], mec='black')
plt.fill_between(time[1:], uddotmean[1:], uddotmean[1:] + sigma*np.sqrt(uddotvar[1:]), color=tableau20[7], alpha=0.5)
plt.fill_between(time[1:], uddotmean[1:], uddotmean[1:] - sigma*np.sqrt(uddotvar[1:]), color=tableau20[7], alpha=0.5)

plt.xlabel('time [s]')
#plt.ylabel('response')
plt.legend(loc='best', frameon=False)
plt.savefig('smd-galerkin-one-sigma.pdf',
            bbox_inches='tight', pad_inches=0.05)

plt.figure()
sigma = 2.0
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.plot(time, umean    , '-', label='${E}[{u}(t)]\pm 2\cdot{S}[{u}(t)]$', mew=mew, ms=markerSize, color=tableau20[2], mec='black')
plt.fill_between(time, umean, umean + sigma*np.sqrt(uvar), color=tableau20[3], alpha=0.5)
plt.fill_between(time, umean, umean - sigma*np.sqrt(uvar), color=tableau20[3], alpha=0.5)

plt.plot(time, udotmean , '-', label='${E}[\dot{u}(t)]\pm 2\cdot{S}[\dot{u}(t)]$', mew=mew, ms=markerSize, color=tableau20[4], mec='black')
plt.fill_between(time, udotmean, udotmean + sigma*np.sqrt(udotvar), color=tableau20[5], alpha=0.5)
plt.fill_between(time, udotmean, udotmean - sigma*np.sqrt(udotvar), color=tableau20[5], alpha=0.5)

plt.plot(time[1:], uddotmean[1:], '-', label='${E}[\ddot{u}(t)]\pm 2\cdot{S}[\ddot{u}(t)]$', mew=mew, ms=markerSize, color=tableau20[6], mec='black')
plt.fill_between(time[1:], uddotmean[1:], uddotmean[1:] + sigma*np.sqrt(uddotvar[1:]), color=tableau20[7], alpha=0.5)
plt.fill_between(time[1:], uddotmean[1:], uddotmean[1:] - sigma*np.sqrt(uddotvar[1:]), color=tableau20[7], alpha=0.5)

plt.xlabel('time [s]')
#plt.ylabel('response')
plt.legend(loc='best', frameon=False)
plt.savefig('smd-galerkin-two-sigma.pdf',
            bbox_inches='tight', pad_inches=0.05)


plt.figure()
sigma = 3.0
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.plot(time, umean    , '-', label='${E}[{u}(t)]\pm 3\cdot{S}[{u}(t)]$', mew=mew, ms=markerSize, color=tableau20[2], mec='black')
plt.fill_between(time, umean, umean + sigma*np.sqrt(uvar), color=tableau20[3], alpha=0.5)
plt.fill_between(time, umean, umean - sigma*np.sqrt(uvar), color=tableau20[3], alpha=0.5)

plt.plot(time, udotmean , '-', label='${E}[\dot{u}(t)]\pm 3\cdot{S}[\dot{u}(t)]$', mew=mew, ms=markerSize, color=tableau20[4], mec='black')
plt.fill_between(time, udotmean, udotmean + sigma*np.sqrt(udotvar), color=tableau20[5], alpha=0.5)
plt.fill_between(time, udotmean, udotmean - sigma*np.sqrt(udotvar), color=tableau20[5], alpha=0.5)

plt.plot(time[1:], uddotmean[1:], '-', label='${E}[\ddot{u}(t)]\pm 3\cdot{S}[\ddot{u}(t)]$', mew=mew, ms=markerSize, color=tableau20[6], mec='black')
plt.fill_between(time[1:], uddotmean[1:], uddotmean[1:] + sigma*np.sqrt(uddotvar[1:]), color=tableau20[7], alpha=0.5)
plt.fill_between(time[1:], uddotmean[1:], uddotmean[1:] - sigma*np.sqrt(uddotvar[1:]), color=tableau20[7], alpha=0.5)

plt.xlabel('time [s]')
#plt.ylabel('response')
plt.legend(loc='best', frameon=False)
plt.savefig('smd-galerkin-three-sigma.pdf',
            bbox_inches='tight', pad_inches=0.05)
