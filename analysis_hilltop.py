import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import csv

config = {
    'font' : 'serif',
    'fontweight' : 'bold',
    'size' : 20
}
config1 = {
    'font' : 'serif',
    'size' : 16
}
config2 = {
    'family' : 'Arial',
    'size' : 12
}

## VARIABLES AND DATA (using E-FOLDS instead of time and supernatural units: M_planck=1, c=1, h_b=1)
N_lambda = 1000
lam_values = np.logspace(-10,-2,N_lambda)
N_efolds = 8
N_values = np.linspace(60,88,N_efolds)
ns_probe = np.empty((N_efolds,N_lambda))
r_probe = np.empty((N_efolds,len(lam_values)))

## SLOW-ROLL APPROXIMATION (Impositions: N variable, eps=1 (end of inflation))
datos_1 = {}
x_1 = {}
y_1 = {}
with open('un_sigma.txt','r') as file:
    reader = csv.reader(file,delimiter=';')
    datos_1 = list(reader)
for linea,sublista in enumerate(datos_1):
    for indice,valor in enumerate(sublista):
        if indice == 0:
            x_1[linea] = float(valor)
        else:
            y_1[linea] = float(valor)
X1 = np.array(list(x_1.values()))
Y1 = np.array(list(y_1.values()))
datos_2 = {}
x_2 = {}
y_2 = {}
with open('dos_sigma.txt','r') as file:
    reader = csv.reader(file,delimiter=';')
    datos_2 = list(reader)
for linea,sublista in enumerate(datos_2):
    for indice,valor in enumerate(sublista):
        if indice == 0:
            x_2[linea] = float(valor)
        else:
            y_2[linea] = float(valor)
X2 = np.array(list(x_2.values()))
Y2 = np.array(list(y_2.values()))

plt.figure(1)
plt.scatter(X1,Y1,color='orange',alpha=1,linewidth=0.8,label='Planck + ACT (68% CL)')
plt.scatter(X2,Y2,color='navajowhite',alpha=0.6,linewidth=0.8,label='Planck + ACT (95% CL)')
for j,N in enumerate(N_values):
    N_label = 'N = '+str(N)
    for i,lam in enumerate(lam_values):
        coef_equation_end = [1,4*np.sqrt(2),0,0,-1/lam]
        roots = np.roots(coef_equation_end)
        phi_end = np.float64([r.real for r in roots if np.isclose(r.imag, 0) & (r.real>0)][0])

        f_efolds_limit = lambda t: np.abs(np.abs(1/16*(phi_end)**2 + 1/(16*lam*phi_end**2) - 1/16*t**2 - 1/(16*lam*t**2)) - N)
        efolds_limit = minimize_scalar(f_efolds_limit, bounds=(0,phi_end))
        phi_star = efolds_limit.x
        
        ns_probe[j][i] = 1 - 16/phi_star**2*(5+3/(lam*phi_star**4))/(1/(lam*phi_star**4)-1)**2
        r_probe[j][i] = 512*lam**2*phi_star**6/(1-lam*phi_star**4)**2
    plt.plot(ns_probe[j],r_probe[j],linewidth=3,label=N_label)
plt.xlabel(r'$n_s$',config1)
plt.ylabel(r'$r$',config1)
plt.ylim((0,0.12))
# plt.title('Slow-roll approximations in $n_s$-$r$ plane',config)
plt.legend(prop=config2)
plt.tick_params(axis='both', which='major', labelsize=13.5)
plt.savefig('ns_r_plane.pdf',bbox_inches='tight')

plt.show()