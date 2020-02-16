import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from pspace import PSPACE
from tacs import TACS, elements

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

class VPLUpdate:
    def __init__(self, elem):
        self.element = elem
        return

    def update(self, vals):
        self.element.mu = vals[0]
        return
    
# Define an element in TACS using the pyElement feature
class Vanderpol(elements.pyElement):
    def __init__(self, num_disps, num_nodes, mu):
        self.mu = mu
        return

    def getInitConditions(self, index, X, v, dv, ddv):
        '''Define the initial conditions'''
        v[0] = 1.0
        dv[0] = 1.0
        return

    def addResidual(self, index, time, X, v, dv, ddv, res):
        '''Add the residual of the governing equations'''
        res[0] += ddv[0] - self.mu*(1.0-v[0]*v[0])*dv[0] + v[0]
        return    

    def addJacobian(self, index, time, alpha, beta, gamma, X, v, dv, ddv, res, mat):
        '''Add the Jacobian of the governing equations'''
        res[0] += ddv[0] - self.mu*(1.0-v[0]*v[0])*dv[0] + v[0]        
        mat[0] += gamma - beta*self.mu*(1.0-v[0]*v[0]) \
                   + alpha*(1 + 2*self.mu*v[0]*dv[0]*(1-v[0]*v[0]))
        return
    
if __name__ == '__main__':
    # Create instance of user-defined element
    num_nodes = 1
    num_disps = 1
    mu = 1.0
    vpl = Vanderpol(num_nodes, num_disps, mu)

    pfactory = PSPACE.PyParameterFactory()
    y1 = pfactory.createNormalParameter(mu=1.0, sigma=0.25, dmax=6)
    #y1 = pfactory.createExponentialParameter(mu=1.0, beta=0.25, dmax=3)
    #y1 = pfactory.createUniformParameter(a=0.5, b=1.5, dmax=6)
    pc = PSPACE.PyParameterContainer(basis_type=1)
    pc.addParameter(y1)
    pc.initialize()

    callback = VPLUpdate(vpl)
    vpl = PSPACE.PyStochasticElement(vpl, pc, callback)
    
    # Add user-defined element to TACS
    comm = MPI.COMM_WORLD
    assembler = TACS.Assembler.create(comm=comm,
                                      varsPerNode=num_disps*pc.getNumBasisTerms(),
                                      numOwnedNodes=num_nodes,
                                      numElements=1)    
    conn = np.array([0], dtype=np.intc)
    ptr = np.array([0, 1], dtype=np.intc)    
    assembler.setElementConnectivity(ptr, conn)
    assembler.setElements([vpl])
    assembler.initialize()
    
    # Create Integrator
    t0 = 0.0
    dt = 0.01
    num_steps = 1000
    tf = num_steps*dt
    order = 2
    integrator = TACS.BDFIntegrator(assembler, t0, tf, num_steps, order)
    integrator.setPrintLevel(2)
    integrator.integrate()

    time, umean, udotmean, uddotmean, uvar, udotvar, uddotvar = moments(integrator,
                                                                        num_steps,
                                                                        pc.getNumBasisTerms())

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
    plt.savefig('vpl-galerkin-multivariate-expectation.pdf',
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
    plt.savefig('vpl-galerkin-multivariate-variance.pdf',
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
    plt.savefig('vpl-galerkin-one-sigma.pdf',
                bbox_inches='tight', pad_inches=0.05)

    plt.figure()
    sigma = 2.0
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
    plt.savefig('vpl-galerkin-two-sigma.pdf',
                bbox_inches='tight', pad_inches=0.05)


    plt.figure()
    sigma = 3.0
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    plt.plot(time, umean    , '-', label='${E}[{u}(t)]\pm 1\cdot{S}[{u}(t)]$', mew=mew, ms=markerSize, color=tableau20[2], mec='black')
    plt.fill_between(time, umean, umean + sigma*np.sqrt(uvar), color=tableau20[3], alpha=0.5)
    plt.fill_between(time, umean, umean - sigma*np.sqrt(uvar), color=tableau20[3], alpha=0.5)
    
    plt.plot(time, udotmean , '-', label='${E}[\dot{u}(t)]\pm 2\cdot{S}[\dot{u}(t)]$', mew=mew, ms=markerSize, color=tableau20[4], mec='black')
    plt.fill_between(time, udotmean, udotmean + sigma*np.sqrt(udotvar), color=tableau20[5], alpha=0.5)
    plt.fill_between(time, udotmean, udotmean - sigma*np.sqrt(udotvar), color=tableau20[5], alpha=0.5)
    
    plt.plot(time[1:], uddotmean[1:], '-', label='${E}[\ddot{u}(t)]\pm 3\cdot{S}[\ddot{u}(t)]$', mew=mew, ms=markerSize, color=tableau20[6], mec='black')
    plt.fill_between(time[1:], uddotmean[1:], uddotmean[1:] + sigma*np.sqrt(uddotvar[1:]), color=tableau20[7], alpha=0.5)
    plt.fill_between(time[1:], uddotmean[1:], uddotmean[1:] - sigma*np.sqrt(uddotvar[1:]), color=tableau20[7], alpha=0.5)
    
    plt.xlabel('time [s]')
    #plt.ylabel('response')
    plt.legend(loc='best', frameon=False)
    plt.savefig('vpl-galerkin-three-sigma.pdf',
                bbox_inches='tight', pad_inches=0.05)
