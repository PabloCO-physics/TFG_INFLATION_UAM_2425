import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.integrate import solve_ivp
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

## VARIABLES AND DATA (using natural units: M_planck=1, c=1, h_b=1)
N = [0,80] # limits of e-folds
conver = 3.808339152e56 # from Mpl to Mpc^-1
As = 2.107e-9
s_As = 0.025e-9
ns = 0.9690
s_ns = 0.0035
r = 0.032

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
print('The end of inflation ocurrs at: '+str(Ne_limit)+', so phi(N=N*) = '+str(phi(Ne_star)))

# More background solutions
eta = lambda t: 3 + (3 - eps(t))/phi_dot(t)*4/phi(t)
phi_dot2 = lambda t: (eps(t) - 3)*phi_dot(t) + (eps(t) - 3)*4/phi(t)
phi_dot3 = lambda t: 3*(eps(t) - 1)*phi_dot2(t) + phi_dot(t)*(phi_dot2(t)*4/phi(t) + (eps(t) - 3)*12/(phi(t)**2) + (3 - eps(t))*16/(phi(t)**2))
eps_dot = lambda t: phi_dot(t)*phi_dot2(t)
eta_dot = lambda t: phi_dot(t)*phi_dot2(t) + (phi_dot2(t)/phi_dot(t))**2 - phi_dot3(t)/phi_dot(t)


## NORMALIZATION (Ds(k=0.05) = As)
lam_values = np.linspace(1e-14,6e-14,1000) # proporcionality constant from potencial function (approximated analyticly to 3.47e-14)
Ds_probe = np.empty(len(lam_values))
for i,lam in enumerate(lam_values):
    H = lambda t: np.sqrt(lam*phi(t)**4/(3-eps(t))) # Units of Mpl
    a_0 = 0.05/(H(Ne_star)*conver*np.exp(Ne_star)) # Units of Mpc^-1/Mpc^-1
    a = lambda t: a_0*np.exp(t)

    K = 0.05/conver # Reference mode 0.05 Mpc^-1 (Units of Mpl)
    Ki = K/(2*np.pi*25) # Inicial conditions mode
    f_integration_start = lambda t: np.abs(a(t)*H(t)-Ki)
    integration_start = minimize_scalar(f_integration_start)
    Ne_start = integration_start.x
    f_horizon_crossing = lambda t: np.abs(K/(a(t)*H(t))-1)
    horizon_crossing = minimize_scalar(f_horizon_crossing)
    horcross = horizon_crossing.x

    # R Perturbation
    U_0 = [1.0,0.0,0.0,-K/Ki] # initial conditions for uk_r, uk_i, duk_r, duk_i (Units of 1/sqrt(2K))
    def uPert_ode (t,y):
        d1_ur = y[2]
        d1_vr = - (1 - eps(t))*y[2] - (np.power(K/(a(t)*H(t)),2) + (eps(t)-eta(t)+1)*(eta(t)-2) + eta_dot(t) - eps_dot(t))*y[0]
        d1_ui = y[3]
        d1_vi = - (1 - eps(t))*y[3] - (np.power(K/(a(t)*H(t)),2) + (eps(t)-eta(t)+1)*(eta(t)-2) + eta_dot(t) - eps_dot(t))*y[1]
        return [d1_ur,d1_ui,d1_vr,d1_vi]
    uPert_sol = solve_ivp (uPert_ode, [Ne_start,horcross+5], U_0, dense_output=True)
    ur = lambda t: uPert_sol.sol(t)[0]
    ui = lambda t: uPert_sol.sol(t)[1]
    U = lambda t: np.power(2*K,-0.5)*np.sqrt(ur(t)**2 + ui(t)**2) # Units of Mpl^-1/2
    Ds_probe[i] = (K**3/(2*np.pi**2)*(U(horcross+5)/(a(horcross+5)*np.sqrt(2*eps(horcross+5))))**2)
f_normalization = np.abs(Ds_probe-As)
with open('lambda_quartic.txt','w') as archivo:
    archivo.write(f"{repr(lam_values[np.argmin(f_normalization)])}\n")
    archivo.write(f"{repr(Ds_probe[np.argmin(f_normalization)])}\n")
np.savetxt("Ds_quartic.txt", Ds_probe)

# Ds_probe = np.loadtxt("Ds_quartic.txt") # Once we have found the best agreement, it is not necessary to calculate the results every time

plt.figure(1)
horizontal = As*np.ones(len(lam_values))
horizontal1 = (As+2*s_As)*np.ones(len(lam_values))
horizontal2 = (As-2*s_As)*np.ones(len(lam_values))
plt.plot(lam_values*1e14,Ds_probe*1e9,linewidth=3,label='Slow-roll')
plt.plot(lam_values*1e14,horizontal1*1e9,linestyle='--',color='r',linewidth=3,label=r'$n_s \pm 2\sigma$')
plt.plot(lam_values*1e14,horizontal2*1e9,linestyle='--',color='r',linewidth=3)
plt.plot(lam_values*1e14,horizontal*1e9,linestyle='--',color='k',linewidth=3)
plt.xlabel(r'Parámetro $\lambda \quad \left(10^{-14}\right)$',config1)
plt.xscale('log')
plt.xlim((3,4.5))
plt.xticks(np.arange(3, 5, 0.5))
plt.ylim((1.6,2.7))
plt.ylabel(r'Amplitud $\Delta_s^2 \quad \left(10^{-9}\right)$',config1)
plt.tight_layout()
ax = plt.gca()
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.xaxis.get_major_formatter().set_scientific(False)
ax.xaxis.get_major_formatter().set_useOffset(False)
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.xaxis.get_major_formatter().set_scientific(False)
ax.xaxis.get_major_formatter().set_useOffset(False)
# plt.title(r'Approximation of scalar power spectrum at $k_{*} = 0.05\ Mpc^{-1}$',config)
plt.legend(prop=config2)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig('scalar_amplitude.pdf',bbox_inches='tight')

plt.figure(2)
phix = np.linspace(20,35,100)
horizontal = ns*np.ones(len(phix))
horizontal1 = (ns+2*s_ns)*np.ones(len(phix))
horizontal2 = (ns-2*s_ns)*np.ones(len(phix))
plt.plot(phix,1-24/phix**2,linewidth=3,label='Slow-roll')
plt.plot(phix,horizontal1,linestyle='--',color='r',linewidth=3,label=r'$n_s \pm 2\sigma$')
plt.plot(phix,horizontal2,linestyle='--',color='r',linewidth=3)
plt.plot(phix,horizontal,linestyle='--',color='k',linewidth=3)
plt.xlabel(r'Campo inflatón $\phi \quad (M_{pl})$',config1)
plt.ylabel(r'Índice espectral escalar $n_s$',config1)
# plt.title('Evolution of scalar spectral index',config)
plt.legend(prop=config2)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig('spectral_index_probe.pdf',bbox_inches='tight')

plt.show()