#!/usr/bin/env python3

# execute: python energy_analysis.py test0_CIC/diags

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import m_u, epsilon_0, mu_0, e, m_e
import yt

m_D = 2.0141*m_u - m_e

# this will be the name of the plot file
fn = sys.argv[1]
print('fn = ', fn)

ds = yt.load(fn)
print(ds.field_list)

ds.index
ad = ds.all_data()

ele_weight = ad[("electrons", "particle_weight")].to_ndarray().squeeze()
ele_momx = ad[("electrons", "particle_momentum_x")].to_ndarray().squeeze()
ele_posx = ad[("electrons", "particle_position_x")].to_ndarray().squeeze()
ele_theta = ad[("electrons", "particle_theta")].to_ndarray().squeeze()
print("ele_weight.shape = ",ele_weight.shape)
print("ele_weight.max = ",np.max(ele_weight))
print("ele_weight.min = ",np.min(ele_weight))

#print("sum_weight = ",np.sum(ele_weight))

time = ds.current_time.to_value()
time_ns = time*1.0e9
Nx, Nz, Ny = ds.domain_dimensions
xmin, zmin, ymin = ds.domain_left_edge.v
Lx, Lz, Ly = ds.domain_width.v
xgrid = xmin + Lx / Nx * (0.5 + np.arange(Nx))
data = ds.covering_grid(level = 0, left_edge = ds.domain_left_edge, dims = ds.domain_dimensions)
Nppc_array = data[("boxlib", "part_per_cell")].to_ndarray().squeeze()
nume_array = data[("boxlib", "num_electrons")].to_ndarray().squeeze()
enexe_array = data[("boxlib", "enex_electrons")].to_ndarray().squeeze()
eneye_array = data[("boxlib", "eney_electrons")].to_ndarray().squeeze()
eneze_array = data[("boxlib", "enez_electrons")].to_ndarray().squeeze()
uxe_array = data[("boxlib", "ux_electrons")].to_ndarray().squeeze()
uye_array = data[("boxlib", "uy_electrons")].to_ndarray().squeeze()
uze_array = data[("boxlib", "uz_electrons")].to_ndarray().squeeze()
numD_array = data[("boxlib", "num_deuterium")].to_ndarray().squeeze()
enexD_array = data[("boxlib", "enex_deuterium")].to_ndarray().squeeze()
eneyD_array = data[("boxlib", "eney_deuterium")].to_ndarray().squeeze()
enezD_array = data[("boxlib", "enez_deuterium")].to_ndarray().squeeze()
uxD_array = data[("boxlib", "ux_deuterium")].to_ndarray().squeeze()
uyD_array = data[("boxlib", "uy_deuterium")].to_ndarray().squeeze()
uzD_array = data[("boxlib", "uz_deuterium")].to_ndarray().squeeze()
divE = data[("boxlib", "divE")].to_ndarray().squeeze()
rho = data[("boxlib", "rho")].to_ndarray().squeeze()
#ele_weight = data[("electrons", "particle_weight")].value
Er_array = data[("boxlib", "Er")].to_ndarray().squeeze()
Et_array = data[("boxlib", "Et")].to_ndarray().squeeze()
Ez_array = data[("boxlib", "Ez")].to_ndarray().squeeze()
Br_array = data[("boxlib", "Br")].to_ndarray().squeeze()
Bt_array = data[("boxlib", "Bt")].to_ndarray().squeeze()
Bz_array = data[("boxlib", "Bz")].to_ndarray().squeeze()

# Compute electron temperature
vx_mean = np.divide(uxe_array, nume_array, out=np.zeros_like(uxe_array), where=nume_array!=0)
vy_mean = np.divide(uye_array, nume_array, out=np.zeros_like(uye_array), where=nume_array!=0)
vz_mean = np.divide(uze_array, nume_array, out=np.zeros_like(uze_array), where=nume_array!=0)
v2x = np.divide(enexe_array, nume_array, out=np.zeros_like(enexe_array), where=nume_array!=0)
v2y = np.divide(eneye_array, nume_array, out=np.zeros_like(eneye_array), where=nume_array!=0)
v2z = np.divide(eneze_array, nume_array, out=np.zeros_like(eneze_array), where=nume_array!=0)
Tele_X = m_e/e * (2.0 * v2x - vx_mean**2)
Tele_Y = m_e/e * (2.0 * v2y - vy_mean**2)
Tele_Z = m_e/e * (2.0 * v2z - vz_mean**2)
Tele = (Tele_X + Tele_Y + Tele_Z)/3.0

# Compute deuterium temperature
vxD_mean = np.divide(uxD_array, numD_array, out=np.zeros_like(uxD_array), where=numD_array!=0)
vyD_mean = np.divide(uyD_array, numD_array, out=np.zeros_like(uyD_array), where=numD_array!=0)
vzD_mean = np.divide(uzD_array, numD_array, out=np.zeros_like(uzD_array), where=numD_array!=0)
v2xD = np.divide(enexD_array, numD_array, out=np.zeros_like(enexD_array), where=numD_array!=0)
v2yD = np.divide(eneyD_array, numD_array, out=np.zeros_like(eneyD_array), where=numD_array!=0)
v2zD = np.divide(enezD_array, numD_array, out=np.zeros_like(enezD_array), where=numD_array!=0)
Tion_X = m_D/e * (2.0 * v2xD - vxD_mean**2)
Tion_Y = m_D/e * (2.0 * v2yD - vyD_mean**2)
Tion_Z = m_D/e * (2.0 * v2zD - vzD_mean**2)
Tion = (Tion_X + Tion_Y + Tion_Z)/3.0


print("sum(Nppc) = ",np.sum(Nppc_array))
dx = xgrid[1]-xgrid[0]
dV = 2.0*np.pi*xgrid*dx

print("Nppc_array.shape = ", Nppc_array.shape)
print("Lx = ", Lx)
print("Nx = ", Nx)
print("time  = ", time)

print("mean Nppc = ", np.mean(Nppc_array[Nppc_array !=0]))

#max_val = np.max(Ex_diff_nodal)
#min_val = np.min(Ex_diff_nodal)
#cbar_val = min(max_val,abs(min_val))
#print("cbar_val = ", cbar_val)
#levels = np.linspace(-cbar_val, cbar_val, 100)

plt.rcParams.update({
    "font.size": 24,
    "axes.labelsize": 24,
    "axes.titlesize": 26,
    "legend.fontsize": 22,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22
})

fig, ax = plt.subplots(2, 1, figsize=(12,10), constrained_layout=True)

ax[0].plot(xgrid*100, nume_array/dV, label='electron')
ax[0].plot(xgrid*100, numD_array/dV, label='ion')
ax[0].set_xlabel(r'$r [cm]$')
ax[0].set_ylabel(r'$number$')
ax[0].set_title(f'species density (t = {time_ns:.3e} ns)')
ax[0].legend()
#ax[0].set_ylim(0,3.5e23)

#ax[1].plot(xgrid*100, 0*xgrid + 2.1e6)
ax[1].plot(xgrid*100, Tele, label='electron')
ax[1].plot(xgrid*100, Tion, label='ion')
ax[1].set_xlabel(r'$r [cm]$')
ax[1].set_ylabel(r'$temperature [eV]$')
ax[1].set_title(f'species temperature (t = {time_ns:.3e} ns)')
ax[1].legend()
#ax[1].set_ylim(0,2000)
plt.show()

fig2, ax2 = plt.subplots(2, 1, figsize=(12,10), constrained_layout=True)

ax2[0].plot(xgrid*100, Nppc_array)
ax2[0].set_xlabel(r'$r [cm]$')
ax2[0].set_ylabel(r'$number$')
ax2[0].set_title(f'particles per cell (t = {time_ns:.3e} ns)')
ax2[0].legend()

ax2[1].plot(xgrid*100, epsilon_0*divE, label='ep0*divE')
ax2[1].plot(xgrid*100, rho, linestyle='--', label='rho')
ax2[1].set_xlabel(r'$r [cm]$')
ax2[1].set_ylabel(r'$\delta\rho$')
ax2[1].set_title(f'charge conservation (t = {time_ns:.3e} ns)')
ax2[1].legend()
plt.show()

fig3, ax3 = plt.subplots(2, 1, figsize=(12,10), constrained_layout=True)

#ax3[0].plot(xgrid*100, 2.0*np.pi*xgrid*Bt_array/mu_0)
ax3[0].plot(xgrid*100, Br_array)
ax3[0].plot(xgrid*100, Bt_array)
ax3[1].plot(xgrid*100, Bz_array)
ax2[0].set_xlabel(r'$r [cm]$')
ax2[0].set_ylabel(r'$B_r and B_t [T]$')
ax2[0].legend()

ax3[1].plot(xgrid*100, Er_array)
ax3[1].plot(xgrid*100, Et_array)
ax3[1].plot(xgrid*100, Ez_array)
ax3[1].set_xlabel(r'$r [cm]$')
ax3[1].set_ylabel(r'$E_r [V/m] and E_t [V/m]$')
plt.show()

#ax[1].plot(xgrid, Nppc_array)
#ax[1].set_title('particles per cell')
#ax[1].plot(xgrid, nume_array/Nppc_array)
#ax[1].plot(ele_theta)
#ax[1].set_title('particle theta')
#ax[1].set_title('divE')
#ax[1].plot(ele_weight)
#ax[1].set_title('electron particle weights')
#ax[1].set_xlabel(r'$x [cm]$')
#ax[1].set_xlim(20,35)

#plt.tight_layout()
plt.show()
