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

## Potential
lam_values = np.logspace(-2,0,3)
phix = np.linspace(0,5,1000)
plt.figure()
for i,lam in enumerate(lam_values):
    V = (1 - lam*phix**4)**2
    plt.plot(phix,V,linewidth=3)
plt.ylim((0,2.5))
plt.xlim((0,4))
plt.xlabel(r'Campo inflatón $\phi$',config1)
plt.ylabel(r'Potencial $\frac{V}{V_0}$',config1)
# plt.title('Squared quartic hilltop potential',config)
plt.legend([r'$\lambda = $'+str(lam_values[0]),r'$\lambda = $'+str(lam_values[1]),r'$\lambda = $'+str(lam_values[2])],prop=config2)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig('examples_potential.pdf',bbox_inches='tight')

## Variables and data (using supernatural units: M_planck=1, c=1, h_b=1)
with open('lambda_hilltop.txt','r') as archivo:
    lineas = archivo.readlines()
    for l,value in enumerate(lineas):
        if l==0:
            lam = np.float64(value)
        else:
            V0 = np.float64(value)
conver = 3.808339152e56 # from Mpl to Mpc^-1

# Potential
V = lambda t: V0*(1 - lam*t**4)**2 # Units Mpl^4
V_fd = lambda t: 8*lam*V0*t**3*(lam*t**4 - 1) # Units Mpl^3
V_sd = lambda t: 8*lam*V0*t**2*(7*lam*t**4 - 3) # Units Mpl^2

## Background equation
def background_ode (t,y):
    # inflaton field (x) and its derivative (y)
    d1_x = y[1]
    d1_y = - (3 - 0.5*y[1]**2)*y[1] - (3 - 0.5*y[1]**2)*V_fd(y[0])/V(y[0])
    return [d1_x,d1_y]

## Phase diagram, field and Hubble's radius beyond end of inflation
coef_equation_end = [1,4*np.sqrt(2),0,0,-1/lam]
roots = np.roots(coef_equation_end)
phi_limit_appr = np.float64([r.real for r in roots if np.isclose(r.imag, 0) & (r.real>0)][0])
f_efolds_limit = lambda t: np.abs(np.abs(1/16*(phi_limit_appr)**2 + 1/(16*lam*phi_limit_appr**2) - 1/16*t**2 - 1/(16*lam*t**2))-70)
efolds_limit = minimize_scalar(f_efolds_limit, bounds=(0,phi_limit_appr))
phi_0 = efolds_limit.x
Ne_limit_appr = np.abs(np.abs(1/16*(phi_limit_appr)**2 + 1/(16*lam*phi_limit_appr**2) - 1/16*phi_0**2 - 1/(16*lam*phi_0**2)))
N = [0,Ne_limit_appr+10] # Range of e-folds
background_conditions = [phi_0,-V_fd(phi_0)/V(phi_0)]

background_sol = solve_ivp (background_ode, N, background_conditions, dense_output=True, method='BDF', rtol=1e-4, atol=1e-5, max_step=0.001) # To obtain evolution beyond eps=1: method='BDF', rtol=1e-4, atol=1e-5, max_step=0.001
phi = lambda t: background_sol.sol(t)[0]
phi_dot = lambda t: background_sol.sol(t)[1]
eps = lambda t: 0.5*phi_dot(t)**2
f_inflation_limit = lambda t: np.abs(eps(t)-1.0)
inflation_limit = minimize_scalar(f_inflation_limit, bounds=(Ne_limit_appr-1,Ne_limit_appr+2))
Ne_limit = inflation_limit.x
Ne_star = Ne_limit - 60
print('The end of inflation occurs at N = '+str(Ne_limit)+' with phi (Mpl) = '+str(phi(Ne_limit)))

Ne_bey = np.linspace(0,Ne_limit+2,10000)
x = np.linspace(phi_0,phi(Ne_limit)+0.4,100)
y = -V_fd(x)/V(x)
Y = np.linspace(-4,4,100)
X = np.empty(len(Y))
for i in range(len(Y)):
    X[i] = phi(Ne_limit)

plt.figure(1)
line1 = plt.plot(phi(Ne_bey),phi_dot(Ne_bey),color='b',linewidth=3,label='Numerical')
line2 = plt.plot(x,y,linestyle='--',color='r',linewidth=3,label='cuasi-de Sitter')
line3 = plt.plot(X,Y,linestyle='--',color='g',linewidth=3,label='Slow-roll limit')
plt.xlabel(r'Inflaton field $\phi$',config1)
plt.ylabel(r'Derivative of inflaton field $\frac{d\phi}{dN}$',config1)
# plt.title(r'Phase diagram beyond $\epsilon = 1$ (in e-folds)',config)
plt.xlim(phi_0-0.5,phi(Ne_limit)+2)
plt.ylim(-3,3)
plt.legend(prop=config2)
plt.tick_params(axis='both',which='major',labelsize=14)
plt.savefig('phase_diagrama_beyond.pdf',bbox_inches='tight')

Y = np.linspace(phi_0,phi(Ne_limit)+2,100)
X = np.empty(len(Y))
for i in range(len(Y)):
    X[i] = Ne_limit-Ne_star
plt.figure(4)
plt.plot(Ne_bey-Ne_star,phi(Ne_bey),linewidth=3,label='Numerical')
plt.plot(X,Y,linestyle='--',color='g',linewidth=3,label='Slow-roll limit')
plt.xlabel(r'N e-folds',config1)
plt.ylabel(r'Inflaton field $\phi$',config1)
# plt.title(r'Evolution of $\phi$ beyond $\epsilon = 1$ (in e-folds)',config)
plt.xlim(45,Ne_limit+2.5-Ne_star)
plt.ylim(phi(45+Ne_star)-0.5,phi(Ne_limit)+1.5)
plt.legend(prop=config2)
plt.tick_params(axis='both',which='major',labelsize=14)
plt.savefig('evolution_field.pdf',bbox_inches='tight')

H = lambda t: np.sqrt(V(phi(t))/(3-eps(t))) # Units of Mpl
a_0 = 0.05/(H(Ne_star)*conver*np.exp(Ne_star)) # Units of Mpc^-1/Mpc^-1
a = lambda t: a_0*np.exp(t)
plt.figure(7)
plt.plot(Ne_bey-Ne_star,np.log(np.abs(1/(a(Ne_bey)*H(Ne_bey)*conver))),linewidth=3)
plt.xlabel('N e-folds',config1)
plt.ylabel(r'Natural logarithm $\left(\frac{Mpc_{-1}}{aH}\right)$',config1)
# plt.title('Evolution of Hubble radius',config)
plt.xlim((45,Ne_limit+2.3-Ne_star))
plt.ylim(np.log(1/(a(Ne_limit)*H(Ne_limit)*conver))-1,np.log(1/(a(45+Ne_star)*H(45+Ne_star)*conver))+1)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig('Hubble_radius.pdf',bbox_inches='tight')

energy_density = lambda t: 0.5*H(t)**2*phi_dot(t)**2 + V(phi(t))
pression = lambda t: 0.5*H(t)**2*phi_dot(t)**2 - V(phi(t))
w = lambda t: pression(t)/energy_density(t)
print('The end of inflation coincides with w = '+str(w(Ne_limit))) # inflation limit takes place when w=-1/3 for the first time
w_inflation = quad(lambda t: 2/3*eps(t)-1,0,Ne_limit)[0]/Ne_limit
w_beyond = quad(lambda t: 2/3*eps(t)-1,Ne_limit,Ne_limit+2)[0]/2
print('The average value of w during inflation is: '+str(w_inflation)+', while the average value 2 e-folds after the end of inflation is: '+str(w_beyond))
fig, ax_main = plt.subplots(num=10)
ax_main.plot(Ne_bey-Ne_star,w(Ne_bey),Ne_bey-Ne_star,2/3*eps(Ne_bey)-1,linewidth=3)
ax_main.set_xlabel('N e-folds',config1)
ax_main.set_ylabel(r'Ecuación de estado $\omega$',config1)
# ax_main.set_title('Evolution of the equation of state',config)
ax_main.legend([r'$\frac{p}{\rho}$',r'$\frac{2}{3} \epsilon - 1$'],prop=config2)
ax_main.tick_params(axis='both', which='major', labelsize=14)
ax_inset = fig.add_axes([0.25,0.25,0.375,0.375])
ax_inset.plot(Ne_bey-Ne_star,w(Ne_bey),Ne_bey-Ne_star,2/3*eps(Ne_bey)-1,linewidth=3)
ax_inset.set_xlim(Ne_limit-Ne_star,Ne_limit+2.2-Ne_star)
plt.savefig('state_eq.pdf',bbox_inches='tight')

plt.show()