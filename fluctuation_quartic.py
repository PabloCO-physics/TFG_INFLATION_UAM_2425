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
    'family' : 'Arial',
    'size' : 17
}

## VARIABLES AND DATA (using E-FOLDS instead of time and supernatural units: M_planck=1, c=1, h_b=1)
with open('lambda_quartic.txt','r') as archivo:
    lineas = archivo.readlines()
    lam = np.float64(lineas[0])
N = [0,75] # limits of e-folds
conver = 3.808339152e56 # from Mpl to Mpc^-1

## BACKGROUND
# Initial conditions:
phi_0 = 23.5
y_0 = [phi_0,-4/phi_0]

# Solving equation: d2(phi) + (3-eps)*d1(phi) + Mpl^2*(3-eps)*d_phi(ln V) = 0
def background_ode (t,y):
    # inflaton field (x) and its derivative (y)
    d1_x = y[1]
    d1_y = - (3 - 0.5*y[1]**2)*y[1] - (3 - 0.5*y[1]**2)*4/y[0]
    return [d1_x,d1_y]
background_sol = solve_ivp (background_ode, N, y_0, dense_output=True)
phi = lambda t: background_sol.sol(t)[0]
phi_dot = lambda t: background_sol.sol(t)[1]
eps = lambda t: 0.5*phi_dot(t)**2
f_inflation_limit = lambda t: np.abs(eps(t)-1.0)
inflation_limit = minimize_scalar(f_inflation_limit)
Ne_limit = inflation_limit.x
Ne_star = Ne_limit - 60
print('The end of inflation ocurrs at: '+str(Ne_limit))

# More background solutions
eta = lambda t: 3 + (3 - eps(t))/phi_dot(t)*4/phi(t)
phi_dot2 = lambda t: (eps(t) - 3)*phi_dot(t) + (eps(t) - 3)*4/phi(t)
phi_dot3 = lambda t: 3*(eps(t) - 1)*phi_dot2(t) + phi_dot(t)*(phi_dot2(t)*4/phi(t) + (eps(t) - 3)*12/(phi(t)**2) + (3 - eps(t))*16/(phi(t)**2))
eps_dot = lambda t: phi_dot(t)*phi_dot2(t)
eta_dot = lambda t: phi_dot(t)*phi_dot2(t) + (phi_dot2(t)/phi_dot(t))**2 - phi_dot3(t)/phi_dot(t)
H = lambda t: np.sqrt(lam*phi(t)**4/(3-eps(t))) # Units of Mpl
a_0 = 0.05/(H(Ne_star)*conver*np.exp(Ne_star)) # Units of Mpc^-1/Mpc^-1
a = lambda t: a_0*np.exp(t)

## SCALAR PERTURBATIONS
K = 0.05/conver # Mode example 0.05 Mpc^-1 (Units of Mpl)
Ki = K/(2*np.pi*20) # Inicial conditions mode
U_0 = [1.0,0.0,0.0,-K/Ki] # initial conditions for uk_r, uk_i, duk_r, duk_i (Units of 1/sqrt(2K))
f_integration_start = lambda t: np.abs(a(t)*H(t)-Ki)
integration_start = minimize_scalar(f_integration_start)
Ne_start = integration_start.x
Ne_per = np.linspace(Ne_start,Ne_limit,100000)
S = lambda t: K/(a(t)*H(t)) # Horizon scale
f_horizon_crossing = lambda t: np.abs(K/(a(t)*H(t))-1)
horizon_crossing = minimize_scalar(f_horizon_crossing)
horcross = horizon_crossing.x

# delta_phi perturbation
# initial conditions for (dphi)k_r, (dphi)k_i, d(dphi)k_r, d(dphi)k_i (Units of 1/(a(Ne_start)*sqrt(2K)))
def dphi_ode (t,y):
    d1_ph_r = y[2]
    d1_dph_r = - (3 - eps(t))*y[2] - y[0]*(np.power(K/(a(t)*H(t)),2) + (3-eps(t))*12/(phi(t)**2) - 2*eps(t)*(3 + eps(t) - 2*eta(t)))
    d1_ph_i = y[3]
    d1_dph_i = - (3 - eps(t))*y[3] - y[1]*(np.power(K/(a(t)*H(t)),2) + (3-eps(t))*12/(phi(t)**2) - 2*eps(t)*(3 + eps(t) - 2*eta(t)))
    return [d1_ph_r,d1_ph_i,d1_dph_r,d1_dph_i]
dphi_sol = solve_ivp (dphi_ode, [Ne_start,Ne_limit], U_0, dense_output=True)
dphi_r = lambda t: dphi_sol.sol(t)[0]
dphi_i = lambda t: dphi_sol.sol(t)[1]
dphi = lambda t: 1/a(Ne_start)*np.sqrt(dphi_r(t)**2 + dphi_i(t)**2) # (Units of 1/sqrt(2K))
# Analytic solution
k_cond = K/(a(Ne_start)*H(0))
k_cub = (a(0)*H(0)/K)**3
M = np.array([
    [np.cos(k_cond)*k_cond - np.sin(k_cond) , - (np.cos(k_cond) + np.sin(k_cond)*k_cond) , k_cub*(np.cos(k_cond) + np.sin(k_cond)*k_cond) , k_cub*(np.sin(k_cond) - np.cos(k_cond)*k_cond)] ,
    [np.cos(k_cond) + np.sin(k_cond)*k_cond , (np.cos(k_cond)*k_cond - np.sin(k_cond)) , k_cub*(np.cos(k_cond)*k_cond - np.sin(k_cond)) , k_cub*(np.cos(k_cond) + np.sin(k_cond)*k_cond)] ,
    [k_cond**2*np.sin(k_cond) , k_cond**2*np.cos(k_cond) , - k_cond**2*k_cub*np.cos(k_cond) , - k_cond**2*k_cub*np.sin(k_cond)] ,
    [-k_cond**2*np.cos(k_cond) , k_cond**2*np.sin(k_cond) , k_cond**2*k_cub*np.sin(k_cond) , - k_cond**2*k_cub*np.cos(k_cond)] ])
coef = np.linalg.solve(M,np.array(U_0))
dphi_an_r = lambda t: coef[0]*(np.cos(K/(a(t)*H(0)))*K/(a(t)*H(0)) - np.sin(K/(a(t)*H(0)))) - coef[1]*(np.cos(K/(a(t)*H(0))) + np.sin(K/(a(t)*H(0)))*K/(a(t)*H(0))) + coef[2]*k_cub*(np.cos(K/(a(t)*H(0))) + np.sin(K/(a(t)*H(0)))*K/(a(t)*H(0))) + coef[3]*k_cub*(np.sin(K/(a(t)*H(0))) - np.cos(K/(a(t)*H(0)))*K/(a(t)*H(0)))
dphi_an_i = lambda t: coef[0]*(np.cos(K/(a(t)*H(0))) + np.sin(K/(a(t)*H(0)))*K/(a(t)*H(0))) + coef[1]*(np.cos(K/(a(t)*H(0)))*K/(a(t)*H(0)) - np.sin(K/(a(t)*H(0)))) + coef[2]*k_cub*(np.cos(K/(a(t)*H(0)))*K/(a(t)*H(0)) - np.sin(K/(a(t)*H(0)))) + coef[3]*k_cub*(np.sin(K/(a(t)*H(0)))*K/(a(t)*H(0)) + np.cos(K/(a(t)*H(0))))
dphi_an = lambda t: 1/a(Ne_start)*np.sqrt(dphi_an_r(t)**2 + dphi_an_i(t)**2) # (Units of 1/sqrt(2K))

plt.figure(1)
plt.plot(Ne_per,np.log10(dphi(Ne_per)*a(Ne_start)),Ne_per,np.log10(dphi_an(Ne_per)*a(Ne_start)),linewidth=2)
plt.xlabel('N e-folds',config1)
plt.ylabel(r'Decimal logarithm of $\left(\frac{\delta\phi_K}{a_i \cdot (2K)^{-1/2}}\right)$',config1)
plt.title(r'Evolution of inflaton fluctuation $\delta\phi_K$',config)
plt.legend(['Exact eq. (numerical)','de Sitter eq. (analytic)'],prop=config2)
plt.tick_params(axis='both', which='major', labelsize=12.5)
plt.savefig('Solution_deltaphi.pdf',bbox_inches='tight')

# Mukhanov-Saski's variable: u
def uPert_ode (t,y):
    d1_ur = y[2]
    d1_vr = - (1 - eps(t))*y[2] - (np.power(K/(a(t)*H(t)),2) + (eps(t)-eta(t)+1)*(eta(t)-2) + eta_dot(t) - eps_dot(t))*y[0]
    d1_ui = y[3]
    d1_vi = - (1 - eps(t))*y[3] - (np.power(K/(a(t)*H(t)),2) + (eps(t)-eta(t)+1)*(eta(t)-2) + eta_dot(t) - eps_dot(t))*y[1]
    return [d1_ur,d1_ui,d1_vr,d1_vi]
uPert_sol = solve_ivp (uPert_ode, [Ne_start,Ne_limit], U_0, dense_output=True)
ur = lambda t: uPert_sol.sol(t)[0]
ui = lambda t: uPert_sol.sol(t)[1]
U = lambda t: np.sqrt(ur(t)**2 + ui(t)**2) # Units of 1/sqrt(2K)
# Analytic solution
A = np.array([
    [-3/4*(np.cos(k_cond) - np.sin(k_cond)/k_cond) , 0 , 2*(np.sin(k_cond) + np.cos(k_cond)/k_cond) , 0] ,
    [0 , -3/4*(np.cos(k_cond) - np.sin(k_cond)/k_cond) , 0 , 2*(np.sin(k_cond) + np.cos(k_cond)/k_cond)] ,
    [-3/4*(np.cos(k_cond) - (1-k_cond**2)*np.sin(k_cond)/k_cond) , 0 , 2*(np.sin(k_cond) + (1-k_cond**2)*np.cos(k_cond)/k_cond) , 0] ,
    [0 , -3/4*(np.cos(k_cond) - (1-k_cond**2)*np.sin(k_cond)/k_cond) , 0 , 2*(np.sin(k_cond) + (1-k_cond**2)*np.cos(k_cond)/k_cond)] ])
coef_u = np.linalg.solve(A,np.array(U_0))
u_an_r = lambda t: -3/4*coef_u[0]*(np.cos(K/(a(t)*H(0))) - np.sin(K/(a(t)*H(0)))*a(t)*H(0)/K) + 2*coef_u[2]*(np.sin(K/(a(t)*H(0))) + np.cos(K/(a(t)*H(0)))*a(t)*H(0)/K)
u_an_i = lambda t: -3/4*coef_u[1]*(np.cos(K/(a(t)*H(0))) - np.sin(K/(a(t)*H(0)))*a(t)*H(0)/K) + 2*coef_u[3]*(np.sin(K/(a(t)*H(0))) + np.cos(K/(a(t)*H(0)))*a(t)*H(0)/K)
U_an = lambda t: np.sqrt(u_an_r(t)**2 + u_an_i(t)**2) # Units of 1/sqrt(2K)

plt.figure(2)
plt.plot(Ne_per,np.log10(U(Ne_per)),Ne_per,np.log10(U_an(Ne_per)),Ne_per,np.log10(a(Ne_per)*dphi(Ne_per)),Ne_per,np.log10(a(Ne_per)*dphi_an(Ne_per)),linewidth=2)
plt.xlabel('N e-folds',config1)
plt.ylabel(r'Decimal logarithm of $\left(\frac{u_K}{(2K)^{-1/2}}\right)$',config1)
plt.title(r'Evolution of variable $u_K$',config)
plt.legend(['Exact eq. (numerical)','de Sitter eq. (analytic)',r'Comparison $a\cdot\delta\phi_K$ (numerical)',r'Comparison $a\cdot\delta\phi_K$ (analytic)'],prop=config2,loc='best')
plt.tick_params(axis='both', which='major', labelsize=12.5)
plt.savefig('Solution_u.pdf',bbox_inches='tight')

# R perturbation
Ru = lambda t: U(t)/(np.sqrt(2*K)*a(t)*np.sqrt(2*eps(t))) # Units of Mpl^-3/2
Rr = lambda t: ur(t)/(np.sqrt(2*K)*a(t)*np.sqrt(2*eps(t)))
Ri = lambda t: ui(t)/(np.sqrt(2*K)*a(t)*np.sqrt(2*eps(t)))
Rp = lambda t: dphi(t)/(np.sqrt(2*K)*np.sqrt(2*eps(t))) # Units of Mpl^-3/2
phi_dot_an = lambda t: -4/phi(t)
Ran = lambda t: U_an(t)/(np.sqrt(2*K)*a(t)*phi_dot_an(0)) # Units of Mpl^-3/2

# We denote conformal time tau = 0 as time at which N = Ne_limit
tau = np.empty(len(Ne_per))
f = lambda t: 1/(a(t)*H(t)*conver) # Units of Mpc
for i,x in enumerate(Ne_per):
    resultado, _ = quad(f, Ne_limit, x)
    tau[i] = resultado
print('Conformal time of horizon crossing is: '+str(tau[np.abs(Ne_per - horcross).argmin()]))
print(a(Ne_start))
plt.figure(3)
plt.plot(tau,np.log10(np.abs(Rr(Ne_per)*a(Ne_start)*np.sqrt(2*K))),color='b',linewidth=2.5)
plt.plot(tau,np.log10(np.abs(Ri(Ne_per)*a(Ne_start)*np.sqrt(2*K))),color='r',linewidth=2.5)
plt.xlabel(r'Tiempo conforme $\tau \, (Mpc)$',config1)
plt.ylabel(r'Logaritmo decimal de $\vert \mathcal{R}_K \vert$',config1)
plt.xlim((-1200,10))
# plt.title(r'Evolution of complex variable $\mathcal{R}_K$',config)
plt.legend([r'$\mathbf{Re}(\mathcal{R}_k)$',r'$\mathbf{Im}(\mathcal{R}_k)$'],prop=config2)
plt.tick_params(axis='both', which='major', labelsize=14)
# plt.savefig('solution_R_comp.pdf',bbox_inches='tight')

leng = 100 # Secondary graphics
Y = np.linspace(-1,5,leng)
X = np.empty(leng)
for i in range(leng):
    X[i] = horcross
plt.figure(4)
ax1 = plt.subplot()
ax2 = ax1.twinx()
line1 = ax1.plot(Ne_per,np.log10(Ru(Ne_per)*a(Ne_start)*np.sqrt(2*K)),linewidth=2.5,color='b',label=r'Perturbación $\mathcal{R}_K$ $\left(\frac{u_K}{z}\right)$')
line2 = ax1.plot(Ne_per,np.log10(np.abs(Ran(Ne_per)*a(Ne_start)*np.sqrt(2*K))),linewidth=2.5,color='r',label=r'Perturbación $\mathcal{R}_K$ $\left(\frac{u_K}{z}\right)$ (de Sitter)')
line3 = ax2.plot(X,Y,linestyle='--',color='k',linewidth=2.5,label='Horizonte')
line4 = ax2.plot(Ne_per,S(Ne_per),linestyle='--',linewidth=2.5,color='g',label=r'$K/aH$')
ax1.set_xlabel('N e-folds',config1)
ax1.set_ylabel(r'Logaritmo decimal de $\vert \mathcal{R}_K \vert$',config1)
ax2.set_ylabel(r'Escala $\frac{K}{aH}$',config1)
ax2.set_ylim(-0.15,5)
# plt.title(r'Evolution of curvature fluctuations $\mathcal{R}_K$',config)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)
lines = line1 + line2 + line3 + line4
labels = [l.get_label() for l in lines]
ax1.legend(lines,labels,prop=config2)
# plt.savefig('solution_R.pdf',bbox_inches='tight')

## TENSOR PERTURBATIONS
# h perturbation (sections k,l) (Recordatory: vk_s has the same initial conditions than uk, but the equation is quite different)
def vPert_ode (t,y):
    d1_ur = y[2]
    d1_vr = - (1 - eps(t))*y[2] - (np.power(K/(a(t)*H(t)),2) + eps(t) - 2)*y[0]
    d1_ui = y[3]
    d1_vi = - (1 - eps(t))*y[3] - (np.power(K/(a(t)*H(t)),2) + eps(t) - 2)*y[1]
    return [d1_ur,d1_ui,d1_vr,d1_vi]
vPert_sol = solve_ivp (vPert_ode, [Ne_start,Ne_limit], U_0, dense_output=True)
vr = lambda t: vPert_sol.sol(t)[0]
vi = lambda t: vPert_sol.sol(t)[1]
V = lambda t: np.sqrt(vr(t)**2+vi(t)**2) # Units of 1/sqrt(2K)
h = lambda t: 2/(a(t)*np.sqrt(2*K))*V(t) # Units of Mpl^-3/2
# Analytic solution is the same than the one for uk, so we're going to use that solution
h_an = lambda t: 2/(a(t)*np.sqrt(2*K))*U_an(t) # Units of Mpl^-3/2

plt.figure(5)
ax1 = plt.subplot()
ax2 = ax1.twinx()
line1 = ax1.plot(Ne_per,np.log10(h(Ne_per)*a(Ne_start)*np.sqrt(2*K)),linewidth=2.5,color='b',label=r'Perturbación $h^s_K$ $\left(\frac{2v_K}{a}\right)$')
line2 = ax1.plot(Ne_per,np.log10(h_an(Ne_per)*a(Ne_start)*np.sqrt(2*K)),linewidth=2.5,color='r',label=r'Perturbación $h^s_K$ $\left(\frac{2v_K}{a}\right)$ (de Sitter)')
line3 = ax2.plot(X,Y,linestyle='--',color='k',linewidth=2.5,label='Horizonte')
line4 = ax2.plot(Ne_per,S(Ne_per),linestyle='--',color='g',linewidth=2.5,label=r'$K/aH$')
ax1.set_xlabel('N e-folds',config1)
ax1.set_ylabel(r'Logaritmo decimal $\vert h^s_K \vert$',config1)
ax2.set_ylabel(r'Escala $\frac{K}{aH}$',config1)
ax2.set_ylim(-0.15,5)
# plt.title(r'Evolution of tensorial perturbation $h^s_K$',config)
lines = line1 + line2 + line3 + line4
labels = [l.get_label() for l in lines]
ax1.legend(lines,labels,prop=config2)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)
# plt.savefig('solution_h.pdf',bbox_inches='tight')

plt.show()