import numpy as np
import math

from pspace.orthogonal_polynomials import unit_hermite  as Hhat
from pspace.orthogonal_polynomials import unit_legendre as Phat
from pspace.orthogonal_polynomials import unit_laguerre as Lhat
from pspace.orthogonal_polynomials import hermite  as H
from pspace.orthogonal_polynomials import legendre as P
from pspace.orthogonal_polynomials import laguerre as L

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker

###################################################################
# plot polynomials
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
plt.rcParams['text.latex.preamble'] = r'\usepackage{sfmath}'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'courier'
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['lines.color'] = 'r'

# Make sure everything is within the frame
plt.rcParams.update({'figure.autolayout': True})

# These are the "Colors 20" colors as RGB.
colors = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(colors)):
    r, g, b = colors[i]
    colors[i] = (r / 255., g / 255., b / 255.)

#######################################################################
# Hermite polynomials
#######################################################################

plt.figure()
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
u = np.linspace(-2.0, 2.0, 100, True)
plt.subplots_adjust(right=0.7)
plt.plot(u, H(u,0), '-o', lw = 2, label = '${H}_1$', color = colors[0] , mec='black', markevery = 6)
plt.plot(u, H(u,1), '-o', lw = 2, label = '${H}_2$', color = colors[2] , mec='black', markevery = 6)
plt.plot(u, H(u,2), '-o', lw = 2, label = '${H}_3$', color = colors[4] , mec='black', markevery = 6)
plt.plot(u, H(u,3), '-o', lw = 2, label = '${H}_4$', color = colors[6] , mec='black', markevery = 6)
plt.plot(u, H(u,4), '-o', lw = 2, label = '${H}_5$', color = colors[8] , mec='black', markevery = 6)
plt.plot(u, H(u,5), '-o', lw = 2, label = '${H}_6$', color = colors[10], mec='black', markevery = 6)
plt.ylabel('{H}(z)')
plt.xlabel('z')
plt.legend(bbox_to_anchor=(1.01,0.5), loc="center left", borderaxespad=0)
plt.savefig('hermite-polynomials.pdf', bbox_inches='tight', pad_inches=0.05)

plt.figure()
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
u = np.linspace(-2.0, 2.0, 100, True)
plt.plot(u, Hhat(u,0), '-o', lw = 2, label = '$\widehat{H_1}$', color = colors[0] , mec='black', markevery = 6)
plt.plot(u, Hhat(u,1), '-o', lw = 2, label = '$\widehat{H_2}$', color = colors[2] , mec='black', markevery = 6)
plt.plot(u, Hhat(u,2), '-o', lw = 2, label = '$\widehat{H_3}$', color = colors[4] , mec='black', markevery = 6)
plt.plot(u, Hhat(u,3), '-o', lw = 2, label = '$\widehat{H_4}$', color = colors[6] , mec='black', markevery = 6)
plt.plot(u, Hhat(u,4), '-o', lw = 2, label = '$\widehat{H_5}$', color = colors[8] , mec='black', markevery = 6)
plt.plot(u, Hhat(u,5), '-o', lw = 2, label = '$\widehat{H_6}$', color = colors[10], mec='black', markevery = 6)
plt.ylabel('\widehat{H}(z)')
plt.xlabel('z')
plt.legend(bbox_to_anchor=(1.01,0.5), loc="center left", borderaxespad=0)
plt.savefig('unit-hermite-polynomials.pdf', bbox_inches='tight', pad_inches=0.05)

#######################################################################
# Legendre polynomials
#######################################################################

plt.figure()
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
u = np.linspace(0.0, 1.0, 100, True)
plt.plot(u, P(u,0), '-o', lw = 2, label = '$P_1$', color = colors[0] , mec='black', markevery = 6)
plt.plot(u, P(u,1), '-o', lw = 2, label = '$P_2$', color = colors[2] , mec='black', markevery = 6)
plt.plot(u, P(u,2), '-o', lw = 2, label = '$P_3$', color = colors[4] , mec='black', markevery = 6)
plt.plot(u, P(u,3), '-o', lw = 2, label = '$P_4$', color = colors[6] , mec='black', markevery = 6)
plt.plot(u, P(u,4), '-o', lw = 2, label = '$P_5$', color = colors[8] , mec='black', markevery = 6)
plt.plot(u, P(u,5), '-o', lw = 2, label = '$P_6$', color = colors[10], mec='black', markevery = 6)
plt.ylabel('P(z)')
plt.xlabel('z')
plt.legend(bbox_to_anchor=(1.01,0.5), loc="center left", borderaxespad=0)
plt.savefig('legendre-polynomials.pdf', bbox_inches='tight', pad_inches=0.05)

plt.figure()
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
u = np.linspace(0.0, 1.0, 100, True)
plt.subplots_adjust(right=0.7)
plt.plot(u, Phat(u,0), '-o', lw = 2, label = '$\widehat{P_1}$', color = colors[0] , mec='black', markevery = 6)
plt.plot(u, Phat(u,1), '-o', lw = 2, label = '$\widehat{P_2}$', color = colors[2] , mec='black', markevery = 6)
plt.plot(u, Phat(u,2), '-o', lw = 2, label = '$\widehat{P_3}$', color = colors[4] , mec='black', markevery = 6)
plt.plot(u, Phat(u,3), '-o', lw = 2, label = '$\widehat{P_4}$', color = colors[6] , mec='black', markevery = 6)
plt.plot(u, Phat(u,4), '-o', lw = 2, label = '$\widehat{P_5}$', color = colors[8] , mec='black', markevery = 6)
plt.plot(u, Phat(u,5), '-o', lw = 2, label = '$\widehat{P_6}$', color = colors[10], mec='black', markevery = 6)
plt.ylabel('\widehat{P}(z)')
plt.xlabel('z')
plt.legend(bbox_to_anchor=(1.01,0.5), loc="center left", borderaxespad=0)
plt.savefig('unit-legendre-polynomials.pdf', bbox_inches='tight', pad_inches=0.05)

#######################################################################
# Laguerre polynomials
#######################################################################

plt.figure()
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
u = np.linspace(0.0, 4.0, 100, True)
plt.plot(u, Lhat(u,0)*math.factorial(0), '-o', lw = 2, label = '$L_1$', color = colors[0] , mec='black', markevery = 6)
plt.plot(u, Lhat(u,1)*math.factorial(1), '-o', lw = 2, label = '$L_2$', color = colors[2] , mec='black', markevery = 6)
plt.plot(u, Lhat(u,2)*math.factorial(2), '-o', lw = 2, label = '$L_3$', color = colors[4] , mec='black', markevery = 6)
plt.plot(u, Lhat(u,3)*math.factorial(3), '-o', lw = 2, label = '$L_4$', color = colors[6] , mec='black', markevery = 6)
plt.plot(u, Lhat(u,4)*math.factorial(4), '-o', lw = 2, label = '$L_5$', color = colors[8] , mec='black', markevery = 6)
plt.plot(u, Lhat(u,5)*math.factorial(5), '-o', lw = 2, label = '$L_6$', color = colors[10], mec='black', markevery = 6)
plt.ylabel('L(z)')
plt.xlabel('z')
plt.legend(bbox_to_anchor=(1.01,0.5), loc="center left", borderaxespad=0)
plt.savefig('laguerre-polynomials.pdf', bbox_inches='tight', pad_inches=0.05)

plt.figure()
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
u = np.linspace(0.0, 4.0, 100, True)
plt.subplots_adjust(right=0.7)
plt.plot(u, Lhat(u,0), '-o', lw = 2, label = '$\widehat{L_1}$', color = colors[0] , mec='black', markevery = 6)
plt.plot(u, Lhat(u,1), '-o', lw = 2, label = '$\widehat{L_2}$', color = colors[2] , mec='black', markevery = 6)
plt.plot(u, Lhat(u,2), '-o', lw = 2, label = '$\widehat{L_3}$', color = colors[4] , mec='black', markevery = 6)
plt.plot(u, Lhat(u,3), '-o', lw = 2, label = '$\widehat{L_4}$', color = colors[6] , mec='black', markevery = 6)
plt.plot(u, Lhat(u,4), '-o', lw = 2, label = '$\widehat{L_5}$', color = colors[8] , mec='black', markevery = 6)
plt.plot(u, Lhat(u,5), '-o', lw = 2, label = '$\widehat{L_6}$', color = colors[10], mec='black', markevery = 6)
plt.ylabel('\widehat{L}(z)')
plt.xlabel('z')
plt.legend(bbox_to_anchor=(1.01,0.5), loc="center left", borderaxespad=0)
plt.savefig('unit-laguerre-polynomials.pdf', bbox_inches='tight', pad_inches=0.05)
