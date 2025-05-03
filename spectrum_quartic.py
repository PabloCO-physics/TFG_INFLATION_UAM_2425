import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar, root_scalar

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
background_sol = solve_ivp (background_ode, N, y_0, dense_output=True, rtol=1e-8, atol=1e-9)
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

## POWER SPECTRUM
K_values = np.logspace(np.log10(a(Ne_star-4)*H(Ne_star-4)),np.log10(a(Ne_limit-5)*H(Ne_limit-5)),1000) # Units of Mpl
Ds = np.empty(len(K_values)) # dimensionless power spectrum for scalar perturbations
Dt = np.empty(len(K_values)) # dimensionless power spectrum for tensor perturbations
r = np.empty(len(K_values)) # tensor-to-scalar tensor
Ds_appr = np.empty(len(K_values)) # dimensionless power spectrum for scalar perturbations approximated
Dt_appr = np.empty(len(K_values)) # dimensionless power spectrum for tensor perturbations approximated
r_appr = np.empty(len(K_values)) # tensor-to-scalar tensor approximated
ns = np.empty(len(K_values)) # scalar spectral index
ns_appr = np.empty(len(K_values)) # scalar spectral index approximated

for i,K in enumerate(K_values):
    Ki = K/(2*np.pi*20) # Inicial conditions mode
    f_integration_start = lambda t: np.abs(a(t)*H(t)-Ki)
    integration_start = minimize_scalar(f_integration_start,bounds=(0,Ne_limit))
    Ne_start = integration_start.x
    f_horizon_crossing = lambda t: np.abs(K/(a(t)*H(t))-1)
    horizon_crossing = minimize_scalar(f_horizon_crossing,bounds=(0,Ne_limit))
    horcross = horizon_crossing.x
    if i == 0 or i == len(K_values)-1:
        print(str(horcross))
    
    # R Perturbation
    U_0 = [1.0,0.0,0.0,-K/Ki] # initial conditions for uk_r, uk_i, duk_r, duk_i (Units of 1/sqrt(2K))
    def uPert_ode (t,y):
        d1_ur = y[2]
        d1_vr = - (1 - eps(t))*y[2] - (np.power(K/(a(t)*H(t)),2) + (eps(t)-eta(t)+1)*(eta(t)-2) + eta_dot(t) - eps_dot(t))*y[0]
        d1_ui = y[3]
        d1_vi = - (1 - eps(t))*y[3] - (np.power(K/(a(t)*H(t)),2) + (eps(t)-eta(t)+1)*(eta(t)-2) + eta_dot(t) - eps_dot(t))*y[1]
        return [d1_ur,d1_ui,d1_vr,d1_vi]
    uPert_sol = solve_ivp (uPert_ode, [Ne_start,horcross+3.5], U_0, dense_output=True)
    ur = lambda t: uPert_sol.sol(t)[0]
    ui = lambda t: uPert_sol.sol(t)[1]
    U = lambda t: np.power(2*K,-0.5)*np.sqrt(ur(t)**2+ui(t)**2) # Units of Mpl^-1/2

    # h perturbation (Recordatory: vk_s has the same initial conditions than uk, but the equation is quite different)
    def vPert_ode (t,y):
        d1_ur = y[2]
        d1_vr = - (1 - eps(t))*y[2] - (np.power(K/(a(t)*H(t)),2) + eps(t) - 2)*y[0]
        d1_ui = y[3]
        d1_vi = - (1 - eps(t))*y[3] - (np.power(K/(a(t)*H(t)),2) + eps(t) - 2)*y[1]
        return [d1_ur,d1_ui,d1_vr,d1_vi]
    vPert_sol = solve_ivp (vPert_ode, [Ne_start,horcross+3.5], U_0, dense_output=True)
    vr = lambda t: vPert_sol.sol(t)[0]
    vi = lambda t: vPert_sol.sol(t)[1]
    V = lambda t: np.power(2*K,-0.5)*np.sqrt(vr(t)**2+vi(t)**2) # Units of Mpl^-1/2

    # Tensor-scalar ratio r (section n)
    Ds[i] = K**3/(2*np.pi**2)*(U(horcross+3.5)/(a(horcross+3.5)*np.sqrt(2*eps(horcross+3.5))))**2 # Power spectrum dimensionless for scalar perturbation
    Dt[i] = 2*K**3/(2*np.pi**2)*(2/a(horcross+3.5)*V(horcross+3.5))**2 # Power spectrum dimensionless for tensor perturbation
    r[i] = Dt[i]/Ds[i]
    Ds_appr[i] = 1/2*(H(horcross)/(2*np.pi))**2/eps(horcross)
    Dt_appr[i] = 2*(H(horcross)/np.pi)**2
    r_appr[i] = 16*eps(horcross)
    ns_appr[i] = 1 + 2*eta(horcross) - 4*eps(horcross)
ns = 1 + np.gradient(np.log(Ds),np.log(K_values))

# Saving the results
np.savetxt('delta_s.txt',Ds)
np.savetxt('delta_s_appr.txt',Ds_appr)
np.savetxt('delta_t.txt',Dt)
np.savetxt('delta_t_appr.txt',Dt_appr)
np.savetxt('r.txt',r)
np.savetxt('r_appr.txt',r_appr)
np.savetxt('K_values.txt',K_values*conver)
np.savetxt('ns.txt',ns)
np.savetxt('ns_appr.txt',ns_appr)

# # Once we've found the best agreement, it´s not necessary to calculate the results every time
# Ds = np.loadtxt('delta_s.txt')
# Ds_appr = np.loadtxt('delta_s_appr.txt')
# Dt = np.loadtxt('delta_t.txt')
# Dt_appr = np.loadtxt('delta_t_appr.txt')
# r = np.loadtxt('r.txt')
# r_appr = np.loadtxt('r_appr.txt')
# ns = np.loadtxt('ns.txt')
# ns_appr = np.loadtxt('ns_appr.txt')

plt.figure(1,figsize=(10,6))
ax1 = plt.subplot()
ax2 = ax1.twinx()
line1 = ax1.plot(K_values*conver,Ds,color='b',linewidth=3,label=r'$\Delta_{s}^{2}$ (numérico)')
line2 = ax1.plot(K_values*conver,Ds_appr,color='b',linewidth=3,linestyle='--',label=r'$\Delta_{s}^{2}$ (de Sitter)')
line3 = ax1.plot(K_values*conver,Dt,color='r',linewidth=3,label=r'$\Delta_{t}^{2}$ (numérico)')
line4 = ax1.plot(K_values*conver,Dt_appr,color='r',linewidth=3,linestyle='--',label=r'$\Delta_{t}^{2}$ (de Sitter)')
line5 = ax2.plot(K_values*conver,r,color='g',linewidth=3,label='r (numérico)')
line6 = ax2.plot(K_values*conver,r_appr,linewidth=3,linestyle='--',color='g',label='r (de Sitter)')
ax1.set_xscale('log')
ax2.set_xscale('log')
ax1.set_xlabel(r'Modo de Fourier $k \, \left(Mpc^{-1}\right)$',config1)
ax1.set_ylabel(r'Amplitud del espectro',config1)
ax2.set_ylabel(r'Ratio escalar-tensor $r$',config1)
# plt.title(r'Dimensionless power spectrum and tensor-to-scalar ratio',config)
lines = line1 + line2 + line3 + line4 + line5 + line6
labels = [l.get_label() for l in lines]
lg = ax1.legend(lines,labels,prop=config2,bbox_to_anchor=(1.15,1),loc='upper left')
plt.tight_layout()
ax1.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)
# plt.savefig('power_spectrum_dimensionless.pdf',bbox_extra_artists=(lg,),bbox_inches='tight')

plt.figure(2)
plt.plot(K_values*conver,ns,color='b',linewidth=3,label=r'$n_{s}$ (numérico)')
plt.plot(K_values*conver,ns_appr,color='b',linestyle='--',linewidth=3,label=r'$n_{s}$ (de Sitter)')
plt.xscale('log')
plt.xscale('log')
plt.xlabel(r'Modo de Fourier $k \, \left(Mpc^{-1}\right)$',config1)
plt.ylabel(r'Índice espectral escalar $n_s$',config1)
# plt.title(r'Spectral index $n_s$',config)
plt.legend(prop=config2)
plt.tick_params(axis='both', which='major', labelsize=14)
# plt.savefig('spectral_index.pdf',bbox_inches='tight')

plt.show()