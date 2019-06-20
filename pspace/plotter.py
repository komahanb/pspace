import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import numpy as np
import math

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
plt.rcParams['font.sans-serif'] = 'courier'
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
    

def fmt(x, pos):
    if x > 0:
        return '{0:+.2f}'.format(x)
    elif x < 0:
        return '--{0:.2f}'.format(np.abs(x))
    else:
        return "  0.00"

def plot_jacobian(A, name, cmap= plt.cm.coolwarm, normalize=True, precision=1e-6):

    """
    Customized visualization of jacobian matrices for observing
    sparsity patterns
    """
    
    plt.figure()
    fig, ax = plt.subplots()
    
    if normalize is True:
        plt.imshow(A, interpolation='none', cmap=cmap,
                   norm = mpl.colors.Normalize(vmin=-1.,vmax=1.))
    else:
        plt.imshow(A, interpolation='none', cmap=cmap)        
    plt.colorbar(format=ticker.FuncFormatter(fmt))
    
    ax.spy(A, marker='.', markersize=0,  precision=precision)
    
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')

    xlabels = np.linspace(0, A.shape[0], 5, True, dtype=int)
    ylabels = np.linspace(0, A.shape[1], 5, True, dtype=int)

    plt.xticks(xlabels)
    plt.yticks(ylabels)

    plt.savefig(name, bbox_inches='tight', pad_inches=0.05)
    
    plt.close()

    return

def plot_vector(x, name, cmap= plt.cm.coolwarm, normalize=True, precision=1e-6):

    """
    Customized visualization of jacobian matrices for observing
    sparsity patterns
    """
    n = x.shape[0]
    A = np.zeros((n,n))
    A[:,0] = x[:]
    
    plt.figure()
    fig, ax = plt.subplots()
    
    if normalize is True:
        plt.imshow(A, interpolation='none', cmap=cmap,
                   norm = mpl.colors.Normalize(vmin=-1.,vmax=1.))
    else:
        plt.imshow(A, interpolation='none', cmap=cmap)        
    plt.colorbar(format=ticker.FuncFormatter(fmt))
    
    ax.spy(A, marker='.', markersize=0,  precision=precision)
    
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')

    xlabels = np.linspace(0, A.shape[0], 5, True, dtype=int)
    ylabels = np.linspace(0, A.shape[1], 5, True, dtype=int)

    plt.xticks(xlabels)
    plt.yticks(ylabels)

    plt.savefig(name, bbox_inches='tight', pad_inches=0.05)
    
    plt.close()

    return
