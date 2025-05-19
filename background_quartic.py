import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize_scalar

config = {
    'font' : 'serif',
    'fontweight' : 'bold',
    'size' : 20
}
config1 = {
    'font' : 'serif',
    'size' : 17
}
config2 = {
    'family' : 'serif',
    'size' : 17
}

## Background equation
def background_ode (t,y):
    # inflaton field (x) and its derivative (y)
    d1_x = y[1]
    d1_y = - (3 - 0.5*y[1]**2)*y[1] - (3 - 0.5*y[1]**2)*4/y[0]
    return [d1_x,d1_y]

## Phase diagram, inflaton field, Hubble's radius and eq. of state beyond end of inflation
phi_0 = 23.5
y_0 = [phi_0,-4/phi_0]
Ne_limit_appr = 70
N = [0,Ne_limit_appr+10]

background_sol = solve_ivp (background_ode, N, y_0, dense_output=True, method='BDF', rtol=1e-9, atol=1e-10, max_step=0.001) # To obtain evolution beyond eps=1: method='BDF', rtol=1e-9, atol=1e-10, max_step=0.001
phi = lambda t: background_sol.sol(t)[0]
phi_dot = lambda t: background_sol.sol(t)[1]
eps = lambda t: 0.5*phi_dot(t)**2
f_inflation_limit = lambda t: np.abs(eps(t)-1.0)
inflation_limit = minimize_scalar(f_inflation_limit, bounds=(Ne_limit_appr-2,Ne_limit_appr+2))
Ne_limit = inflation_limit.x
Ne_star = Ne_limit - 60
print('The end of inflation occurs at N = '+str(Ne_limit)+' with phi (Mpl) = '+str(phi(Ne_limit)))

Ne_bey = np.linspace(0,Ne_limit+3.5,10000)
x = np.linspace(1/3,phi_0,100)
y = -4/x
Y = np.linspace(-6,4,100)
X = np.empty(len(Y))
for i in range(len(Y)):
    X[i] = phi(Ne_limit)

plt.figure(1)
line1 = plt.plot(phi(Ne_bey),phi_dot(Ne_bey),color='b',linewidth=3,label='Numérico')
line2 = plt.plot(x,y,linestyle='--',color='r',linewidth=3,label='cuasi-de Sitter')
line3 = plt.plot(X,Y,linestyle='--',color='g',linewidth=3,label='Slow-roll límite')
plt.xlabel(r'Campo inflatón $\phi$',config1)
plt.ylabel(r'Derivada del campo $\frac{d\phi}{dN}$',config1)
# plt.title(r'Phase diagram beyond $\epsilon = 1$ (in e-folds)',config)
plt.xlim(-1.5,23.5)
plt.ylim(-6,4)
plt.legend(prop=config2)
plt.tick_params(axis='both',which='major',labelsize=14)
# plt.savefig('phase_diagram_beyond.pdf',bbox_inches='tight')

Y = np.linspace(-3,12,100)
X = np.empty(len(Y))
for i in range(len(Y)):
    X[i] = Ne_limit-Ne_star
XX = np.linspace(0,Ne_limit-0.1-Ne_star,200)
YY = np.sqrt(phi(Ne_star)**2 - 8*XX)
plt.figure(2)
plt.plot(Ne_bey-Ne_star,phi(Ne_bey),linewidth=3,label='Numérico')
plt.plot(XX,YY,linestyle='--',color='r',linewidth=3,label='cuasi-de Sitter')
plt.plot(X,Y,linestyle='--',color='g',linewidth=3,label='Slow-roll límite')
plt.xlabel(r'N e-folds',config1)
plt.ylabel(r'Campo inflatón $\phi$',config1)
# plt.title(r'Evolution of $\phi$ beyond $\epsilon = 1$ (in e-folds)',config)
plt.xlim(45,Ne_limit+4-Ne_star)
plt.ylim(-1.5,12)
plt.legend(prop=config2)
plt.tick_params(axis='both',which='major',labelsize=14)
# plt.savefig('evolution_field.pdf',bbox_inches='tight')

with open('lambda_quartic.txt','r') as archivo:
    lineas = archivo.readlines()
    lam = np.float64(lineas[0])
conver = 3.808339152e56 # from Mpl to Mpc^-1
H = lambda t: np.sqrt(lam*phi(t)**4/(3-eps(t))) # Units of Mpl
a_0 = 0.05/(H(Ne_star)*conver*np.exp(Ne_star)) # Units of Mpc^-1/Mpc^-1
a = lambda t: a_0*np.exp(t)
plt.figure(3)
plt.plot(Ne_bey-Ne_star,np.log(np.abs(1/(a(Ne_bey)*H(Ne_bey)*conver))),linewidth=3)
plt.xlabel('N e-folds',config1)
plt.ylabel(r'Natural logarithm $\left(\frac{Mpc_{-1}}{aH}\right)$',config1)
# plt.title('Evolution of Hubble radius',config)
plt.xlim((45,Ne_limit+3.8-Ne_star))
plt.ylim(np.log(1/(a(Ne_limit)*H(Ne_limit)*conver))-1,np.log(1/(a(45+Ne_star)*H(45+Ne_star)*conver))+1)
plt.tick_params(axis='both', which='major', labelsize=14)
# plt.savefig('Hubble_radius.pdf',bbox_inches='tight')

energy_density = lambda t: 0.5*H(t)**2*phi_dot(t)**2 + lam*phi(t)**4
pression = lambda t: 0.5*H(t)**2*phi_dot(t)**2 - lam*phi(t)**4
w = lambda t: pression(t)/energy_density(t)
print('The end of inflation coincides with w = '+str(w(Ne_limit))) # inflation limit takes place when w=-1/3 for the first time
w_inflation = quad(lambda t: 2/3*eps(t)-1,0,Ne_limit)[0]/Ne_limit
w_beyond = quad(lambda t: 2/3*eps(t)-1,Ne_limit,Ne_limit+3.5)[0]/3.5
print('The average value of w during inflation is: '+str(w_inflation)+', while the average value 3.5 e-folds after the end of inflation is: '+str(w_beyond))
fig, ax_main = plt.subplots(num=4)
ax_main.plot(Ne_bey-Ne_star,w(Ne_bey),Ne_bey-Ne_star,2/3*eps(Ne_bey)-1,linewidth=3)
ax_main.set_xlabel('N e-folds',config1)
ax_main.set_ylabel(r'Ecuación de estado $\omega$',config1)
# ax_main.set_title('Evolution of the equation of state',config)
ax_main.legend([r'$\frac{p}{\rho}$',r'$\frac{2}{3} \epsilon - 1$'],prop=config2)
ax_main.tick_params(axis='both', which='major', labelsize=14)
ax_inset = fig.add_axes([0.25,0.25,0.375,0.375])
ax_inset.plot(Ne_bey-Ne_star,w(Ne_bey),Ne_bey-Ne_star,2/3*eps(Ne_bey)-1,linewidth=3)
ax_inset.set_xlim(Ne_limit-Ne_star,Ne_limit+3.7-Ne_star)
# plt.savefig('state_eq.pdf',bbox_inches='tight')

plt.figure(5)
plt.plot(Ne_bey-Ne_star,energy_density(Ne_bey))

plt.show()