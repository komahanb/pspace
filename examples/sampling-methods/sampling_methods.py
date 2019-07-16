from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from pspace.core import ParameterFactory, ParameterContainer

N = 30

# Create random parameters
pfactory = ParameterFactory()
xi1 = pfactory.createUniformParameter('xi1', dict(a=0.0, b=1.0), 1)
xi2 = pfactory.createUniformParameter('xi2', dict(a=0.0, b=1.0), 1)

# Add random parameters into a container and initialize
pc = ParameterContainer()
pc.addParameter(xi1)
pc.addParameter(xi2)
pc.initialize()

# Get the Gaussian Quadrature points in 2d
Y = np.zeros((N*N,2))
quadmap = pc.getQuadraturePointsWeights({0:N,1:N})
for q in quadmap.keys():
    yq = quadmap[q]['Y']
    wq = quadmap[q]['W']
    Y[q,0] = yq[0]
    Y[q,1] = yq[1]

######################################################################
# plot results
######################################################################

# Configure 
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'

# Optionally set font to Computer Modern to avoid common missing
# font errors
params = {
  'axes.labelsize'  : 20,
  'legend.fontsize' : 14,
  'xtick.labelsize' : 20,
  'ytick.labelsize' : 20,
  'text.usetex'     : True}
plt.rcParams.update(params)

# Latex math
plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath}']
plt.rcParams['font.sans-serif'] = 'courier'
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['lines.color'] = 'r'

# Make sure everything is within the frame
plt.rcParams.update({'figure.autolayout': True})

# Set marker size
markerSize = 7.0 
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
plt.scatter(Y[:,0], Y[:,1], s=np.pi*3, alpha=1.0)
plt.xlabel(r'$\xi_1$')
plt.ylabel(r'$\xi_2$')
plt.axis([0,1,0,1])
plt.legend(loc='best', frameon=False)
plt.savefig('quadrature-samples.pdf', bbox_inches='tight', pad_inches=0.05)

plt.figure()
xx = np.random.rand(N*N)
yy = np.random.rand(N*N)
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.scatter(xx, yy, s=np.pi*3, alpha=1.0)
plt.xlabel(r'$\xi_1$')
plt.ylabel(r'$\xi_2$')
plt.axis([0,1,0,1])
plt.legend(loc='best', frameon=False)
plt.savefig('random-samples.pdf', bbox_inches='tight', pad_inches=0.05)
