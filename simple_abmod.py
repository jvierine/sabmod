import numpy as np
import matplotlib.pyplot as plt
from pymsis import msis
from datetime import datetime


def run_sabmod(rho_scale=1.0, v0=70e3):
    # -----------------------------
    # constants
    # -----------------------------
    rho_m = 7800.0          # kg/m^3  (iron)
    Cd = 1.0                # drag coefficient
    Ch = 0.5                # heat transfer coefficient
    sigma = 5.670374e-8     # Stefan-Boltzmann
    eps = 0.9

    Lv = 6e6                # J/kg latent heat
    cp = 450.0              # J/kg/K iron
    T0 = 200                # initial temp
    T_atm = 200

#    v0 = 70000.0            # m/s
    theta = np.deg2rad(45)

    m0 = 1e-7               # kg (100 \mu g)
    rho_m = 7800

    # initial radius
    r0 = (3*m0/(4*np.pi*rho_m))**(1/3)

    # altitude grid
    z = np.linspace(150e3, 60e3, 2000)

    # storage
    m = np.zeros_like(z)
    v = np.zeros_like(z)
    T = np.zeros_like(z)
    dmdt = np.zeros_like(z)

    m[0] = m0
    v[0] = v0
    T[0] = T0

    # time step
    dt = 1e-3

    date = datetime(2020,1,1)

    for i in range(len(z)-1):

        alt = z[i]

        # MSIS neutral density
        rho = rho_scale*msis.run(date, 0, 0, alt/1000)[0][0]

        r = (3*m[i]/(4*np.pi*rho_m))**(1/3)
        A = np.pi*r**2

        # drag deceleration, ignore g as it is not important here
        dvdt = -(Cd * rho * v[i]**2 * A)/(2*m[i])

        # heating
        Qdot = 0.5 * Ch * rho * v[i]**3 * A

        # radiation
        Qrad = eps * sigma * 4*np.pi*r**2 * (T[i]**4 - T_atm**4)

        # temperature change
        dTdt = (Qdot - Qrad - Lv*0)/(m[i]*cp)

        # simple ablation law
        if T[i] > 1800:

            dmdt[i] = - Qdot / Lv
#            print("mass loss %f %f"%(dmdt[i],alt))

        else:
#            print("no mass loss %f"%(alt))
            dmdt[i] = 0

        # integrate. don't late mass go to zero, as things will explode!
        m[i+1] = max(m[i] + dmdt[i]*dt, 1e-16)
        v[i+1] = v[i] + dvdt*dt
        T[i+1] = T[i] + dTdt*dt
    return(z,dmdt,v)
import numpy as n
import scipy.interpolate as sint

def vel_sweep():
    # --- Publication settings ---
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.2,
        "figure.dpi": 300,
        "font.family": "serif"
    })

    vels = [32e3,52e3,72e3]#np.linspace(30e3,72e3,num=10)

    plt.figure(figsize=(6,5))
    mvels=[]
    mdmdts=[]
    mzs=[]
    peak_alts=[]
    for v in vels:
        print(v)
        z,dmdt,vpr=run_sabmod(rho_scale=1.0,v0=v)
        peak_alts.append(z[n.argmin(dmdt)]/1e3)
        mvels.append(vpr)
        mdmdts.append(dmdt)
        mzs.append(z)

    # Single column width (~3.4 inches)
    fig, ax = plt.subplots(1, 2, figsize=(2*3.4, 1.2*1.9), constrained_layout=True)

    # --- Left panel ---
    for i in range(len(vels)):
        ax[0].plot(mdmdts[i], mzs[i]/1000,
                label=r"$v_0 = %1.2f$ km s$^{-1}$" % (vels[i]/1e3))
        ax[0].axhline(peak_alts[i], color="black", linewidth=0.8, alpha=0.6)

    ax[0].set_ylim([60,110])
    ax[0].set_xlabel(r"$dm/dt$ (kg s$^{-1}$)")
    ax[0].set_ylabel("Altitude (km)")
    ax[0].set_title("(a)")
    ax[0].legend(frameon=False)

    # --- Right panel ---
    for i in range(len(vels)):
        ax[1].plot(mvels[i]/1e3, mzs[i]/1000)
        ax[1].axhline(peak_alts[i], color="black", linewidth=0.8, alpha=0.6)

    ax[1].set_ylim([60,110])
    ax[1].set_xlabel("Velocity (km s$^{-1}$)")
    ax[1].set_ylabel("Altitude (km)")
    ax[1].set_title("(b)")

    # Save as vector graphic (best for journals)
    plt.savefig("meteor_ablation_single_column.png", bbox_inches="tight")

    plt.savefig("meteor_ablation_single_column.pdf", bbox_inches="tight")
    plt.show()
vel_sweep()

def rho_vel_sweep(gamma=0.5):


    peak_alt_funs=[]
#    rho_scales=[0.7,0.8,0.9,1.0,1.1,1.2,1.3]
    rho_scales=[1.0,gamma]#1.1,1.2,1.3]

    vels = np.linspace(30e3,72e3,num=20)
    all_rhos=[]
    all_vels=[]
    all_peak_alts=[]
    for r in rho_scales:
        print(r)
        peak_alt_this=[]
        for v in vels:
            z,dmdt,v=run_sabmod(rho_scale=r,v0=v)
            peak_alt=z[n.argmin(dmdt)]
            peak_alt_this.append(peak_alt/1e3)
#            all_rhos.append(r)
 #           all_vels.append(v/1e3)
        all_peak_alts.append(peak_alt_this)#/1e3)
        altfun=sint.interp1d(vels/1e3,peak_alt_this)
        peak_alt_funs.append(altfun)
    #for i in range(len(rho_scales)):
        #plt.scatter(vels/1e3,all_peak_alts[i],c=n.repeat(rho_scales[i],len(vels)),vmin=n.min(rho_scales),vmax=n.max(rho_scales))
        #if i ==0:
        #    cb=plt.colorbar()
        #    cb.set_label("rho scale")
#    fig, ax = plt.subplots(1, 1, figsize=(3.4, 1.9), constrained_layout=True)
    # --- Publication style ---
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.figsize": (6.5, 4.5),
        "figure.dpi": 300,
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2,
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral"
    })

    fig, ax = plt.subplots()

    # --- Atmospheric density curves ---
    for i in range(len(rho_scales)):
        if i == 0:
            label = r"$\rho_a$"
            ax.plot(
                vels/1e3,
                peak_alt_funs[i](vels/1e3),
                color="black",
                alpha=0.7,
                zorder=-1,
                label=label
            )

        else:
            label = rf"${rho_scales[i]:1.1f}\rho_a$"
            ax.plot(
                vels/1e3,
                peak_alt_funs[i](vels/1e3),
                "--",
                color="black",
                alpha=0.7,
                zorder=-1,
                label=label
            )


    # --- Velocity shift lines ---
    for vi in range(len(vels)):

        v0 = vels[vi] / 1e3
        peak_alt = all_peak_alts[0][vi]

        v1 = v0 * gamma**(-1/3)

        if vi == 0:
            label = rf"$v_g {rho_scales[1]:1.1f}^{{-1/3}}$"
        else:
            label = None

        ax.plot(
            [v0, v1],
            [peak_alt, peak_alt],
            color="crimson",
            lw=1.8,
            label=label
        )

    # --- Labels and formatting ---
    ax.set_title("Peak Ablation Height")
    ax.set_xlabel("Velocity (km s$^{-1}$)")
    ax.set_ylabel("Peak Ablation Height (km)")

    ax.grid(True, alpha=0.3)

    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig("peak_ablation_height.png", bbox_inches="tight")
    plt.savefig("peak_ablation_height.pdf", bbox_inches="tight")

    plt.show()

rho_vel_sweep()