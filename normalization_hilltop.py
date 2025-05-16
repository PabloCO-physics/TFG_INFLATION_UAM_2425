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
conver = 3.808339152e56 # from Mpl to Mpc^-1
As = 2.107e-9
s_ns = 0.025e-9
ns = 0.9690
s_ns = 0.0035
r = 0.032

## SLOW-ROLL APPROXIMATION (Impositions: N=60, eps=1 (end of inflation))
lam_values = np.logspace(-10,-2,1000)
ns_probe = np.empty(len(lam_values))
r_probe = np.empty(len(lam_values))
phi_end = np.empty(len(lam_values))
phi_star = np.empty(len(lam_values))
for i,lam in enumerate(lam_values):
    coef_equation_end = [1,4*np.sqrt(2),0,0,-1/lam]
    roots = np.roots(coef_equation_end)
    phi_end[i] = np.float64([r.real for r in roots if np.isclose(r.imag, 0) & (r.real>0)][0])

    f_efolds_limit = lambda t: np.abs(np.abs(1/16*(phi_end[i])**2 + 1/(16*lam*phi_end[i]**2) - 1/16*t**2 - 1/(16*lam*t**2))-60)
    efolds_limit = minimize_scalar(f_efolds_limit, bounds=(0,phi_end[i]))
    phi_star[i] = efolds_limit.x
    
    ns_probe[i] = 1 - 16/phi_star[i]**2*(5+3/(lam*phi_star[i]**4))/(1/(lam*phi_star[i]**4)-1)**2
    r_probe[i] = 512*lam**2*phi_star[i]**6/(1-lam*phi_star[i]**4)**2


plt.figure(1)
plt.plot(lam_values,ns_probe,linewidth=3,label='Slow-roll')
horizontal = ns*np.ones(len(lam_values))
horizontal1 = (ns+2*s_ns)*np.ones(len(lam_values))
horizontal2 = (ns-2*s_ns)*np.ones(len(lam_values))
plt.plot(lam_values,horizontal1,linestyle='--',color='r',linewidth=3,label=r'$n_s \pm 2\sigma$')
plt.plot(lam_values,horizontal2,linestyle='--',color='r',linewidth=3)
plt.plot(lam_values,horizontal,linestyle='--',color='k',linewidth=3)
plt.xlabel(r'Parámetro $\lambda$ del modelo hilltop',config1)
plt.xscale('log')
plt.ylabel(r'Índice espectral escalar $n_s$',config1)
# plt.title('Approximation of scalar spectral index',config)
plt.legend(prop=config2)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig('spectral_index.pdf',bbox_inches='tight')

plt.figure(2)
plt.plot(lam_values,r_probe,linewidth=3,label='Slow-roll')
horizontal3 = r*np.ones(len(lam_values))
plt.plot(lam_values,horizontal3,linestyle='--',color='r',linewidth=3,label=r'$r_{max}$')
plt.xlabel(r'Parámetro $\lambda$ del modelo hilltop',config1)
plt.xscale('log')
plt.ylabel(r'Ratio escalar-tensor $r$',config1)
# plt.title('Approximation of tensor-to-scalar ratio',config)
plt.legend(prop=config2)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig('ratio.pdf',bbox_inches='tight')

f_spectral_index = np.abs(ns_probe-0.965) # Example value of ns compatible with r
ns_slowroll = ns_probe[np.argmin(f_spectral_index)]
lam_slowroll = lam_values[np.argmin(f_spectral_index)]
phi_end_sr = phi_end[np.argmin(f_spectral_index)]
phi_asterisk_sr = phi_star[np.argmin(f_spectral_index)]
V0_slowroll = As*256*(np.pi*lam_slowroll)**2*phi_asterisk_sr**6*(3*(1-lam_slowroll*phi_asterisk_sr**4)**2 - 32*lam_slowroll**2*phi_asterisk_sr**6)/(1 - lam_slowroll*phi_asterisk_sr**4)**6
print('Slow-roll: lambda = '+str(lam_slowroll)+', V0 = '+str(V0_slowroll)+', phi_asterisk = '+str(phi_asterisk_sr)+', phi_limit = '+str(phi_end_sr))


## NUMERICAL APPROXIMATION
V0_values = np.logspace(np.log10(V0_slowroll)-3,np.log10(V0_slowroll)+3,25)
lam_values = np.logspace(-6,-4,40)
ns_probe = np.empty((len(V0_values),len(lam_values))) # V0_values rows and lam_values columns
As_probe = np.empty((len(V0_values),len(lam_values)))
r_probe = np.empty((len(V0_values),len(lam_values)))
for i,V0 in enumerate(V0_values):
    for j,lam in enumerate(lam_values):
        V = lambda t: V0*(1 - lam*t**4)**2 # Units Mpl^4
        V_fd = lambda t: 8*lam*V0*t**3*(lam*t**4 - 1) # Units Mpl^3
        V_sd = lambda t: 8*lam*V0*t**2*(7*lam*t**4 - 3) # Units Mpl^2

        coef_equation_end = [1,4*np.sqrt(2),0,0,-1/lam]
        roots = np.roots(coef_equation_end)
        phi_limit_appr = np.float64([r.real for r in roots if np.isclose(r.imag, 0) & (r.real>0)][0])
        f_efolds_limit1 = lambda t: np.abs(np.abs(1/16*(phi_limit_appr)**2 + 1/(16*lam*phi_limit_appr**2) - 1/16*t**2 - 1/(16*lam*t**2))-60)
        efolds_limit1 = minimize_scalar(f_efolds_limit1, bounds=(0,phi_limit_appr))
        phi_start = efolds_limit1.x
        f_efolds_limit2 = lambda t: np.abs(np.abs(1/16*(phi_limit_appr)**2 + 1/(16*lam*phi_limit_appr**2) - 1/16*t**2 - 1/(16*lam*t**2))-70)
        efolds_limit2 = minimize_scalar(f_efolds_limit2, bounds=(0,phi_start))
        phi_0 = efolds_limit2.x
        Ne_limit_appr = np.abs(np.abs(1/16*(phi_limit_appr)**2 + 1/(16*lam*phi_limit_appr**2) - 1/16*phi_0**2 - 1/(16*lam*phi_0**2)))
        N = [0,Ne_limit_appr+10] # Range of e-folds

        background_conditions = [phi_0,-V_fd(phi_0)/V(phi_0)] # Using de Sitter solution (Units of Mpl)
        def background_ode (t,y):
            # inflaton field (x) and its derivative (y) in e-folds
            d1_x = y[1]
            d1_y = - (3 - 0.5*y[1]**2)*y[1] - (3 - 0.5*y[1]**2)*V_fd(y[0])/V(y[0])
            return [d1_x,d1_y]
        background_sol = solve_ivp (background_ode, N, background_conditions, dense_output=True) # To obtain evolution beyond eps=1: method='BDF', rtol=1e-10, atol=1e-10
        phi = lambda t: background_sol.sol(t)[0]
        phi_dot = lambda t: background_sol.sol(t)[1]

        eps = lambda t: 0.5*phi_dot(t)**2
        f_inflation_limit = lambda t: np.abs(eps(t)-1.0)
        inflation_limit = minimize_scalar(f_inflation_limit,bounds=(Ne_limit_appr-2,Ne_limit_appr+2))
        Ne_limit = inflation_limit.x
        Ne_scale = Ne_limit - 60

        eta = lambda t: 3 + (3 - eps(t))/phi_dot(t)*V_fd(phi(t))/V(phi(t))
        phi_dot2 = lambda t: (eps(t) - 3)*phi_dot(t) + (eps(t) - 3)*V_fd(phi(t))/V(phi(t))
        phi_dot3 = lambda t: 3*(eps(t) - 1)*phi_dot2(t) + phi_dot(t)*(phi_dot2(t)*V_fd(phi(t))/V(phi(t)) + (eps(t) - 3)*V_sd(phi(t))/V(phi(t)) + (3 - eps(t))*(V_fd(phi(t))/V(phi(t)))**2)
        eps_dot = lambda t: phi_dot(t)*phi_dot2(t)
        eta_dot = lambda t: phi_dot(t)*phi_dot2(t) + (phi_dot2(t)/phi_dot(t))**2 - phi_dot3(t)/phi_dot(t)
        H = lambda t: np.sqrt(V(phi(t))/(3-eps(t))) # Units of Mpl
        a_0 = 0.05/(H(Ne_scale)*conver*np.exp(Ne_scale)) # Units Mpc^-1/Mpc^-1
        a = lambda t: a_0*np.exp(t)
        
        K_values = np.logspace(np.log10(0.05*10**(-0.01)/conver),np.log10(0.05*10**(0.01)/conver),15)
        Ds_probe = np.empty(len(K_values))
        for m,K in enumerate(K_values):
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
            Ds_probe[m] = (K**3/(2*np.pi**2)*(U(horcross+5)/(a(horcross+5)*np.sqrt(2*eps(horcross+5))))**2) # Power spectrum dimensionless for scalar perturbation

            if m == int((len(K_values)-1)/2+1):
                def vPert_ode (t,y):
                    d1_ur = y[2]
                    d1_vr = - (1 - eps(t))*y[2] - (np.power(K/(a(t)*H(t)),2) + eps(t) - 2)*y[0]
                    d1_ui = y[3]
                    d1_vi = - (1 - eps(t))*y[3] - (np.power(K/(a(t)*H(t)),2) + eps(t) - 2)*y[1]
                    return [d1_ur,d1_ui,d1_vr,d1_vi]
                vPert_sol = solve_ivp (vPert_ode, [Ne_start,horcross+5], U_0, dense_output=True)
                vr = lambda t: vPert_sol.sol(t)[0]
                vi = lambda t: vPert_sol.sol(t)[1]
                Vh = lambda t: np.power(2*K,-0.5)*np.sqrt(vr(t)**2+vi(t)**2) # Units of Mpl^-1/2
                Dt_probe = 2*K**3/(2*np.pi**2)*(2/a(horcross+5)*Vh(horcross+5))**2 # Power spectrum dimensionless for tensor perturbation
        ns_probe[i,j] = 1 + np.gradient(np.log(Ds_probe),np.log(K_values))[int((len(K_values)-1)/2+1)] # 0.05 Mpc^-1 is centered in K_values
        As_probe[i,j] = Ds_probe[int((len(K_values)-1)/2+1)]
        r_probe[i,j] = Dt_probe/Ds_probe[int((len(K_values)-1)/2+1)]
        print(i,j)
ns_deviation = (ns_probe - ns)**2/ns**2
As_deviation = (As_probe - As)**2/As**2
np.savetxt("ns_deviation.txt", ns_deviation)
np.savetxt("As_deviation.txt", As_deviation)
np.savetxt("r_probe.txt", r_probe)
# Once we've found the best agreement, it´s not necessary to calculate the results every time
# ns_deviation = np.loadtxt('ns_deviation.txt', dtype=float)
# As_deviation = np.loadtxt('As_deviation.txt', dtype=float)
# r_probe = np.loadtxt('r_probe.txt', dtype=float)

r_limits = np.where(r_probe < r)
ns_limits = np.where(ns_deviation < 1e-5)
r_del = np.arange(0,r_limits[1][0])
ns_del1 = np.arange(0,ns_limits[1][0])
ns_del2 = np.arange(ns_limits[1][-1],40)
arr_del = np.concatenate((ns_del1,ns_del2)) # r_del is contained in ns_del1
ns_dev_rn = np.delete(ns_deviation,arr_del,axis=1)
As_dev_rn = np.delete(As_deviation,arr_del,axis=1)
r_probe_rn = np.delete(r_probe,arr_del,axis=1)
lam_values_del = np.delete(lam_values,arr_del)
As_limits = np.where(As_dev_rn < 0.75)
fil_del1 = np.arange(As_limits[0][-1]-1,As_dev_rn.shape[0]) # we include -1 because not all columns at the last row satisfy the condition above
fil_del2 = np.arange(0,As_limits[0][0])
fil_del = np.concatenate((fil_del1,fil_del2)) # r_del is contained in ns_del1
ns_dev_del = np.delete(ns_dev_rn,fil_del,axis=0)
As_dev_del = np.delete(As_dev_rn,fil_del,axis=0)
r_probe_del = np.delete(r_probe_rn,fil_del,axis=0)
V0_values_del = np.delete(V0_values,fil_del)

fig, ax = plt.subplots(num=3,figsize=(9,6))
contour = ax.contour(lam_values_del, V0_values_del, ns_dev_del, levels=20, colors='black')  # Líneas de nivel
ax.contourf(lam_values_del, V0_values_del, ns_dev_del, levels=20, cmap='plasma') # Contorno relleno
# ax.set_title(r'Deviation of $n_s$ from CMB observations')
ax.set_xlabel(r'$\lambda$',config1)
ax.set_ylabel(r'$V_0 \, (M_{pl}^4)$',config1)
ax.set_xscale('log')
ax.set_yscale('log')
cbar = plt.colorbar(ax.contourf(lam_values_del, V0_values_del, ns_dev_del, levels=20, cmap='plasma'), ax=ax)  # Barra de colores
plt.tick_params(axis='both', which='major', labelsize=14,  width=2, length=7)
plt.tick_params(axis='both', which='minor', labelsize=12,  width=2, length=4)
cbar.ax.tick_params(labelsize=14)
plt.savefig('ns_contor.pdf',bbox_inches='tight')

fig, ax = plt.subplots(num=4)
contour = ax.contour(lam_values_del, V0_values_del, As_dev_del, levels=10, colors='black')  # Líneas de nivel
ax.contourf(lam_values_del, V0_values_del, As_dev_del, levels=10, cmap='plasma') # Contorno relleno
# ax.set_title(r'Deviation of $\Delta_s$ from CMB observations')
ax.set_xlabel(r'$\lambda$',config1)
ax.set_ylabel(r'$V_0 \, (M_{pl}^4)$',config1)
ax.set_xscale('log')
ax.set_yscale('log')
cbar = plt.colorbar(ax.contourf(lam_values_del, V0_values_del, As_dev_del, levels=10, cmap='plasma'), ax=ax)  # Barra de colores
plt.tick_params(axis='both', which='major', labelsize=14,  width=2, length=7)
plt.tick_params(axis='both', which='minor', labelsize=12,  width=2, length=4)
cbar.ax.tick_params(labelsize=14)
plt.savefig('As_contor.pdf',bbox_inches='tight')

fig, ax = plt.subplots(num=5)
contour = ax.contour(lam_values_del, V0_values_del, r_probe_del, levels=20, colors='black')  # Líneas de nivel
ax.contourf(lam_values_del, V0_values_del, r_probe_del, levels=20, cmap='plasma') # Contorno relleno
# ax.set_title(r'Determination of $r$ from scalar-tensor power spectrums')
ax.set_xlabel(r'$\lambda$',config1)
ax.set_ylabel(r'$V_0 \, (M_{pl}^4)$',config1)
ax.set_xscale('log')
ax.set_yscale('log')
cbar = plt.colorbar(ax.contourf(lam_values_del, V0_values_del, r_probe_del, levels=20, cmap='plasma'), ax=ax)  # Barra de colores
plt.tick_params(axis='both', which='major', labelsize=14,  width=2, length=7)
plt.tick_params(axis='both', which='minor', labelsize=12,  width=2, length=4)
cbar.ax.tick_params(labelsize=14)
plt.savefig('r_contor.pdf',bbox_inches='tight')

combin = As_dev_del*ns_dev_del
combin_mins = np.where(combin < 1e-10)
V0_final = V0_values_del[combin_mins[0]]
lam_final = lam_values_del[combin_mins[1]]
with open('lambda_hilltop.txt','w') as archivo:
    for elem in lam_final:
        archivo.write(f"{elem}\n")
    for elem in V0_final:
        archivo.write(f"{elem}\n")

plt.show()