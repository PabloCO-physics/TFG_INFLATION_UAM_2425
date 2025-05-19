import numpy as np
import matplotlib.pyplot as plt
import csv
from matplotlib.collections import LineCollection

config = {
    'font' : 'serif',
    'fontweight' : 'bold',
    'size' : 20
}
config1 = {
    'font' : 'serif',
    'size' : 15
}
config2 = {
    'family' : 'Arial',
    'size' : 11
}

## VARIABLES AND DATA (using E-FOLDS instead of time and supernatural units: M_planck=1, c=1, h_b=1)
N_efolds = 200
N_values = np.linspace(60,150,N_efolds)
ns_probe = np.empty((N_efolds))
r_probe = np.empty((N_efolds))

## SLOW-ROLL APPROXIMATION (Impositions: N variable, eps=1 (end of inflation))
datos_1 = {}
x_1 = {}
y_1 = {}
with open('act_un_sigma.txt','r') as file:
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
with open('act_dos_sigma.txt','r') as file:
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
datos_3 = {}
x_3 = {}
y_3 = {}
with open('planck_un_sigma.txt','r') as file:
    reader = csv.reader(file,delimiter=';')
    datos_3 = list(reader)
for linea,sublista in enumerate(datos_3):
    for indice,valor in enumerate(sublista):
        if indice == 0:
            x_3[linea] = float(valor)
        else:
            y_3[linea] = float(valor)
X3 = np.array(list(x_3.values()))
Y3 = np.array(list(y_3.values()))
datos_4 = {}
x_4 = {}
y_4 = {}
with open('planck_dos_sigma.txt','r') as file:
    reader = csv.reader(file,delimiter=';')
    datos_4 = list(reader)
for linea,sublista in enumerate(datos_4):
    for indice,valor in enumerate(sublista):
        if indice == 0:
            x_4[linea] = float(valor)
        else:
            y_4[linea] = float(valor)
X4 = np.array(list(x_4.values()))
Y4 = np.array(list(y_4.values()))

fig, ax = plt.subplots()
plt.scatter(X3,Y3,color='blue',alpha=1,linewidth=0.8,label='Planck PR4 (68% CL)')
plt.scatter(X4,Y4,color='skyblue',alpha=0.5,linewidth=0.8,label='Planck PR4 (95% CL)')
plt.scatter(X1,Y1,color='orange',alpha=1,linewidth=0.8,label='Planck + ACT DR6 (68% CL)')
plt.scatter(X2,Y2,color='navajowhite',alpha=0.18,linewidth=0.8,label='Planck + ACT DR6 (95% CL)')
for j,N in enumerate(N_values):
    phi_star = np.sqrt(8*(N-1))
    
    ns_probe[j] = 1 - 24/phi_star**2
    r_probe[j] = 128/phi_star**2

points = np.array([ns_probe, r_probe]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
values = N_values
lc = LineCollection(segments, cmap='inferno_r', norm=plt.Normalize(values.min(), values.max()))
lc.set_array(values)
lc.set_linewidth(4)
ax.add_collection(lc)
plt.colorbar(lc, ax=ax, label=r'$N_{*}$')
plt.xlabel(r'$n_s$',config1)
plt.ylabel(r'$r$',config1)
plt.ylim((0,0.28))
plt.xlim((0.948,0.983))
# plt.title('Slow-roll approximations in $n_s$-$r$ plane',config)
plt.legend(prop=config2,loc='best')
plt.tick_params(axis='both', which='major', labelsize=13)
plt.savefig('ns_r_plane.pdf',bbox_inches='tight')

plt.show()