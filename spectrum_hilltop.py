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


## VARIABLES AND DATA (using supernatural units: M_planck=1, c=1, h_b=1)
with open('lambda_hilltop.txt','r') as archivo:
    lineas = archivo.readlines()
    for l,value in enumerate(lineas):
        if l==0:
            lam= np.float64(value)
        else:
            V0 = np.float64(value)
conver = 3.808339152e56 # from Mpl to Mpc^-1
Nk = 1000
Ds = np.empty(Nk) # dimensionless power spectrum for scalar perturbations
Dt = np.empty(Nk) # dimensionless power spectrum for tensor perturbations
r = np.empty(Nk) # tensor-to-scalar tensor
ns = np.empty(Nk) # scalar spectral index
Ds_appr = np.empty(Nk) # dimensionless power spectrum for scalar perturbations approximated
Dt_appr = np.empty(Nk) # dimensionless power spectrum for tensor perturbations approximated
r_appr = np.empty(Nk) # tensor-to-scalar tensor approximated
ns_appr = np.empty(Nk) # scalar spectral index approximated

## POTENTIAL
V = lambda t: V0*(1 - lam*t**4)**2 # Units Mpl^4
V_fd = lambda t: 8*lam*V0*t**3*(lam*t**4 - 1) # Units Mpl^3
V_sd = lambda t: 8*lam*V0*t**2*(7*lam*t**4 - 3) # Units Mpl^2

## BACKGROUND
coef_equation_end = [1,4*np.sqrt(2),0,0,-1/lam]
roots = np.roots(coef_equation_end)
phi_limit_appr = np.float64([r.real for r in roots if np.isclose(r.imag, 0) & (r.real>0)][0])
f_efolds_limit = lambda t: np.abs(np.abs(1/16*(phi_limit_appr)**2 + 1/(16*lam*phi_limit_appr**2) - 1/16*t**2 - 1/(16*lam*t**2))-70) # 70 efolds imposed
efolds_limit = minimize_scalar(f_efolds_limit, bounds=(0,phi_limit_appr))
phi_0 = efolds_limit.x
N = [0,80] # Range of e-folds
background_conditions = [phi_0,-V_fd(phi_0)/V(phi_0)]

def background_ode (t,y):
    # inflaton field (x) and its derivative (y)
    d1_x = y[1]
    d1_y = - (3 - 0.5*y[1]**2)*y[1] - (3 - 0.5*y[1]**2)*V_fd(y[0])/V(y[0])
    return [d1_x,d1_y]
background_sol = solve_ivp (background_ode, N, background_conditions, dense_output=True, rtol=1e-8, atol=1e-9)
phi = lambda t: background_sol.sol(t)[0]
phi_dot = lambda t: background_sol.sol(t)[1]
eps = lambda t: 0.5*phi_dot(t)**2
f_inflation_limit = lambda t: np.abs(eps(t)-1.0)
inflation_limit = minimize_scalar(f_inflation_limit,bounds=(68,72))
Ne_limit = inflation_limit.x
Ne_star = Ne_limit - 60
print('The end of inflation ocurrs at N = '+str(Ne_limit)+', reaching a field phi = '+str(phi(Ne_limit))+' (eps = '+str(eps(Ne_limit))+')')

# More background solutions
eta = lambda t: 3 + (3 - eps(t))/phi_dot(t)*V_fd(phi(t))/V(phi(t))
phi_dot2 = lambda t: (eps(t) - 3)*phi_dot(t) + (eps(t) - 3)*V_fd(phi(t))/V(phi(t))
phi_dot3 = lambda t: 3*(eps(t) - 1)*phi_dot2(t) + phi_dot(t)*(phi_dot2(t)*V_fd(phi(t))/V(phi(t)) + (eps(t) - 3)*V_sd(phi(t))/V(phi(t)) + (3 - eps(t))*(V_fd(phi(t))/V(phi(t)))**2)
eps_dot = lambda t: phi_dot(t)*phi_dot2(t)
eta_dot = lambda t: phi_dot(t)*phi_dot2(t) + (phi_dot2(t)/phi_dot(t))**2 - phi_dot3(t)/phi_dot(t)
H = lambda t: np.sqrt(V(phi(t))/(3-eps(t))) # Units of Mpl
a_0 = 0.05/(H(Ne_star)*conver*np.exp(Ne_star))
a = lambda t: a_0*np.exp(t)

# POWER SPECTRUM
K_values = np.logspace(np.log10(a(Ne_star-5)*H(Ne_star-5)),np.log10(a(Ne_limit-5)*H(Ne_limit-5)),Nk) # Units of Mpl
for i,K in enumerate(K_values):
    Ki = K/(2*np.pi*20) # Inicial conditions mode
    f_integration_start = lambda t: np.abs(a(t)*H(t)-Ki)
    integration_start = minimize_scalar(f_integration_start,bounds=(0,Ne_limit))
    Ne_start = integration_start.x
    f_horizon_crossing = lambda t: np.abs(K/(a(t)*H(t))-1)
    horizon_crossing = minimize_scalar(f_horizon_crossing,bounds=(0,Ne_limit))
    horcross = horizon_crossing.x
    if i == 0 or i == len(K_values)-1:
        print(horcross, a(horcross)*H(horcross)*conver)

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
    Vh = lambda t: np.power(2*K,-0.5)*np.sqrt(vr(t)**2+vi(t)**2) # Units of Mpl^-1/2

    # Spectrum, tensor-scalar ratio and spectral index
    Ds[i] = K**3/(2*np.pi**2)*(U(horcross+3.5)/(a(horcross+3.5)*np.sqrt(2*eps(horcross+3.5))))**2 # Power spectrum dimensionless for scalar perturbation
    Dt[i] = 2*K**3/(2*np.pi**2)*(2/a(horcross+3.5)*Vh(horcross+3.5))**2 # Power spectrum dimensionless for tensor perturbation
    r[i] = Dt[i]/Ds[i]
    Ds_appr[i] = 1/2*(H(horcross)/(2*np.pi))**2/eps(horcross)
    Dt_appr[i] = 2*(H(horcross)/np.pi)**2
    r_appr[i] = 16*eps(horcross)
    ns_appr[i] = 1 + 2*eta(horcross) - 4*eps(horcross)
ns = 1 + np.gradient(np.log(Ds),np.log(K_values))

np.savetxt('delta_s.txt',Ds)
np.savetxt('delta_s_appr.txt',Ds_appr)
np.savetxt('delta_t.txt',Dt)
np.savetxt('delta_t_appr.txt',Dt_appr)
np.savetxt('r.txt',r)
np.savetxt('r_appr.txt',r_appr)
np.savetxt('K_values.txt',K_values*conver)
np.savetxt('ns.txt',ns)
np.savetxt('ns_appr.txt',ns_appr)
# Once we've found the best agreement, it´s not necessary to calculate the results every time
# Ds = np.loadtxt('delta_s.txt')
# Ds_appr = np.loadtxt('delta_s_appr.txt')
# Dt = np.loadtxt('delta_t.txt')
# Dt_appr = np.loadtxt('delta_t_appr.txt')
# r = np.loadtxt('r.txt')
# r_appr = np.loadtxt('r_appr.txt')
# ns = np.loadtxt('ns.txt')
# ns_appr = np.loadtxt('ns_appr.txt')
# K_values = np.loadtxt('K_values.txt')

plt.figure(1,figsize=(10,6))
ax1 = plt.subplot()
ax2 = ax1.twinx()
line1 = ax1.plot(K_values,Ds*1e9,color='b',linewidth=3,label=r'$\Delta_{s}^{2}$ (numérico)')
line2 = ax1.plot(K_values,Ds_appr*1e9,color='b',linewidth=3,linestyle='--',label=r'$\Delta_{s}^{2}$ (de Sitter)')
line3 = ax1.plot(K_values,Dt*1e9,color='r',linewidth=3,label=r'$\Delta_{t}^{2}$ (numérico)')
line4 = ax1.plot(K_values,Dt_appr*1e9,color='r',linewidth=3,linestyle='--',label=r'$\Delta_{t}^{2}$ (de Sitter)')
line5 = ax2.plot(K_values,r,color='g',linewidth=3,label='r (numérico)')
line6 = ax2.plot(K_values,r_appr,linewidth=3,linestyle='--',color='g',label='r (de Sitter)')
ax1.set_xscale('log')
ax2.set_xscale('log')
ax1.set_xlabel(r'Modo de Fourier $k \, \left(Mpc^{-1}\right)$',config1)
ax1.set_ylabel(r'Amplitud del espectro $(10^{-9})$',config1)
ax2.set_ylabel(r'Ratio escalar-tensor $r$',config1)
# plt.title(r'Dimensionless power spectrum and tensor-to-scalar ratio',config)
lines = line1 + line2 + line3 + line4 + line5 + line6
labels = [l.get_label() for l in lines]
lg = ax1.legend(lines,labels,prop=config2,bbox_to_anchor=(1.15,1),loc='upper left')
plt.tight_layout()
ax1.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)
plt.savefig('spectrum_ratio.pdf',bbox_extra_artists=(lg,),bbox_inches='tight')

plt.figure(2)
plt.plot(K_values,ns,color='b',linewidth=3,label=r'$n_{s}$ (numérico)')
plt.plot(K_values,ns_appr,color='b',linestyle='--',linewidth=3,label=r'$n_{s}$ (de Sitter)')
plt.xscale('log')
plt.xscale('log')
plt.xlabel(r'Modo de Fourier $k \, \left(Mpc^{-1}\right)$',config1)
plt.ylabel(r'Índice espectral escalar $n_s$',config1)
# plt.title(r'Spectral index $n_s$',config)
plt.legend(prop=config2)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig('spectral_index.pdf',bbox_inches='tight')

plt.show()