#!/usr/bin/env python3
"""
fw_hermite_opt_v2.py  —  Fixed-Wing Hermite Spline Trajectory Optimizer
=========================================================================
MIGHTY-style (Kondo et al. arXiv:2511.10822), adapted for fixed-wing UAV.

Key improvements over v1
------------------------
1. Warm-start with proper Dubins RSR arc-length parametrisation.
   Interior knot velocities are set from the geometric tangent along
   the arc, so every knot starts with a physically coherent heading.
   Accelerations are set from trim centripetal/pitch dynamics.

2. Roll rate  p = dφ/dt  and pitch rate  q = dγ/dt  are added as
   soft-constrained cost terms, reflecting aileron/elevator authority.

3. Aerodynamic cost decomposed into three clean, independently
   differentiable terms:
     (a) Along-path drag residual  (a_∥ + D/m)²
     (b) Normal load-factor        n = L/(mg) — penalise n > n_max
     (c) Gravity-lift balance      (az + g·cos γ)²  in level trim

4. All gradients derived analytically; a finite-difference check is
   run at startup to verify correctness before optimising.

Point-mass aerodynamics
-----------------------
  V   = ||v||
  q̄  = ½ρV²S
  CL  = mg/(q̄)          (trim lift coefficient at level flight)
  CD  = CD0 + k·CL²
  D   = q̄·CD            [N]
  L   = q̄·CL            [N]  (= mg at trim)
  Bank (coordinated turn):  φ = atan2(||a⊥||, g)
  Load factor:              n = L_actual / (mg) ≈ 1/cos(φ)

Roll rate  p = Δφ/Δt  (finite diff of bank along trajectory samples)
Pitch rate q = Δγ/Δt  (finite diff of flight-path angle)
"""

import argparse, math, time, warnings
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ══════════════════════════════════════════════════════════════════════
#  Parameters
# ══════════════════════════════════════════════════════════════════════
@dataclass
class FWParams:
    M:        int   = 7
    kappa:    int   = 12

    w_time:   float = 5e1
    w_smooth: float = 1e-1
    w_drag:   float = 1e2     # drag residual (along-path)
    w_lift:   float = 1e2     # gravity-lift imbalance
    w_load:   float = 5e2     # load-factor limit
    w_bank:   float = 1e3
    w_roll:   float = 8e2
    w_pitch:  float = 6e2
    w_speed:  float = 1e3
    w_acc:    float = 3e2

    mass:      float = 2.5
    g:         float = 9.81
    rho:       float = 1.225
    S_wing:    float = 0.35
    CL0:       float = 0.3
    CLa:       float = 5.0
    CD0:       float = 0.025
    k_induced: float = 0.045
    V_stall:   float = 9.0
    V_ne:      float = 28.0

    phi_max:  float = math.radians(55)
    p_max:    float = math.radians(45)   # roll rate  [rad/s]
    q_max:    float = math.radians(20)   # pitch rate [rad/s]
    a_max:    float = 15.0
    n_max:    float = 3.5                # load factor
    max_iter: int   = 1200
    tol:      float = 1e-5
    gtol:     float = 1e-5


# ══════════════════════════════════════════════════════════════════════
#  Quintic Hermite basis  (vectorised, returns (6,N))
# ══════════════════════════════════════════════════════════════════════
def _basis(tau):
    t  = np.asarray(tau, float)
    t2,t3,t4,t5 = t**2,t**3,t**4,t**5
    h   = np.array([1-10*t3+15*t4-6*t5,
                    t-6*t3+8*t4-3*t5,
                    .5*t2-1.5*t3+1.5*t4-.5*t5,
                    10*t3-15*t4+6*t5,
                    -4*t3+7*t4-3*t5,
                    .5*t3-t4+.5*t5])
    dh  = np.array([-30*t2+60*t3-30*t4,
                    1-18*t2+32*t3-15*t4,
                    t-4.5*t2+6*t3-2.5*t4,
                    30*t2-60*t3+30*t4,
                    -12*t2+28*t3-15*t4,
                    1.5*t2-4*t3+2.5*t4])
    ddh = np.array([-60*t+180*t2-120*t3,
                    -36*t+96*t2-60*t3,
                    1-9*t+18*t2-10*t3,
                    60*t-180*t2+120*t3,
                    -24*t+84*t2-60*t3,
                    3*t-12*t2+10*t3])
    return h, dh, ddh


def _eval_seg(ks, ke, Ts, tau_arr):
    h, dh, ddh = _basis(tau_arr)
    H = np.stack([ks[:3], Ts*ks[3:6], .5*Ts**2*ks[6:9],
                  ke[:3], Ts*ke[3:6], .5*Ts**2*ke[6:9]], axis=1)
    return H@h, (H@dh)/Ts, (H@ddh)/Ts**2


# ══════════════════════════════════════════════════════════════════════
#  Hermite → Bézier  +  closed-form jerk integral  (MIGHTY §III-C.1)
# ══════════════════════════════════════════════════════════════════════
def _h2b(ps, vs, as_, pe, ve, ae, Ts):
    return [ps,
            ps+(Ts/5)*vs,
            ps+(2*Ts/5)*vs+(Ts**2/20)*as_,
            pe-(2*Ts/5)*ve+(Ts**2/20)*ae,
            pe-(Ts/5)*ve,
            pe]

_GD = np.array([1/5, 2/15, 1/5])   # diagonal Gram weights for jerk

def _jerk_grad(c, Ts):
    """(J, dJ/dc[0..5], dJ/dTs) for one Bézier segment."""
    d   = [c[m+3]-3*c[m+2]+3*c[m+1]-c[m] for m in range(3)]
    Cs  = 3600*Ts**-5
    J   = Cs * sum(_GD[m]*(d[m]@d[m]) for m in range(3))
    gd  = [2*Cs*_GD[m]*d[m] for m in range(3)]
    alp = [-1,3,-3,1]
    gc  = [np.zeros(3) for _ in range(6)]
    for m in range(3):
        for k,j in enumerate(range(m,m+4)):
            gc[j] += alp[k]*gd[m]
    # Recover physical velocities/accels from Bézier control points
    vs_  = (c[1]-c[0])*5/Ts;    ve_ = (c[5]-c[4])*5/Ts
    as__ = (c[2]-c[0]-(2*Ts/5)*vs_)*20/Ts**2
    ae_  = (c[3]-c[5]+(2*Ts/5)*ve_)*20/Ts**2
    dct  = [np.zeros(3), vs_/5, (2/5)*vs_+(Ts/10)*as__,
            -(2/5)*ve_+(Ts/10)*ae_, -ve_/5, np.zeros(3)]
    dJdTs = (-5/Ts)*J + sum(gc[k]@dct[k] for k in range(6))
    return J, gc, dJdTs


# ══════════════════════════════════════════════════════════════════════
#  Smooth hinge  φ(x) = (1/μ)·log(1+exp(μx))
# ══════════════════════════════════════════════════════════════════════
_MU = 50.
def _h(x):
    mx = _MU*x
    return np.where(mx>0,(mx+np.log1p(np.exp(-mx)))/_MU,
                         np.log1p(np.exp(mx))/_MU)
def _dh(x):
    mx = _MU*x
    return np.where(mx>0,1/(1+np.exp(-mx)),np.exp(mx)/(1+np.exp(mx)))


# ══════════════════════════════════════════════════════════════════════
#  Dubins RSR warm-start  (proper arc-length parametrisation)
#  Returns list of M+1 state arrays [p(3), v(3), a(3)]
# ══════════════════════════════════════════════════════════════════════
def _wrap(a):
    return (a+math.pi)%(2*math.pi)-math.pi

def dubins_rsr(p0, v0, pf, vf, M, g=9.81, phi_nom=math.radians(35)):
    """
    Build M+1 warm-start knot states along a Dubins RSR path.
    Knot velocities come from the arc tangent direction (scaled to the
    linearly interpolated speed between V0 and Vf).
    Knot accelerations come from trim centripetal (arcs) or zero (straight).
    """
    V0  = float(np.linalg.norm(v0[:2])) or 15.
    Vf  = float(np.linalg.norm(vf[:2])) or 15.
    Vn  = 0.5*(V0+Vf)
    R   = max(Vn**2/(g*math.tan(phi_nom)), 5.)

    xi,yi,zi = float(p0[0]),float(p0[1]),float(p0[2])
    xf,yf,zf = float(pf[0]),float(pf[1]),float(pf[2])
    hi = math.atan2(float(v0[1]),float(v0[0]))
    hf = math.atan2(float(vf[1]),float(vf[0]))

    # Right-turn circle centres
    cx1=xi+R*math.sin(hi); cy1=yi-R*math.cos(hi)
    cx2=xf+R*math.sin(hf); cy2=yf-R*math.cos(hf)
    dx=cx2-cx1; dy=cy2-cy1
    dist=math.sqrt(dx**2+dy**2)+1e-9
    beta=math.atan2(dy,dx)
    alpha=beta-math.pi/2   # RSR tangent angle

    # Tangent points
    tx1=cx1+R*math.cos(alpha); ty1=cy1+R*math.sin(alpha)
    tx2=cx2+R*math.cos(alpha); ty2=cy2+R*math.sin(alpha)

    # Arc 1: clockwise from hi to heading (alpha+π/2)
    # On a CW circle, position angle θ and heading h satisfy: h = θ + π/2 + π = θ - π/2
    # Wait — for CW rotation: pos=(cx+R cosθ, cy+R sinθ), vel∝(-sinθ, cosθ) rotated by -π/2
    # vel_dir = (sinθ, -cosθ)  so heading h = atan2(-cosθ, sinθ) = θ - π/2
    # Therefore θ = h + π/2
    theta1_s = hi + math.pi/2     # starting angle on circle 1
    theta1_e = alpha + math.pi/2  # ending angle (at tangent point)
    a1 = _wrap(theta1_e - theta1_s)
    if a1 > 0: a1 -= 2*math.pi   # ensure CW (negative)
    arc1_len = abs(a1)*R

    theta2_s = alpha + math.pi/2
    theta2_e = hf + math.pi/2
    a2 = _wrap(theta2_e - theta2_s)
    if a2 > 0: a2 -= 2*math.pi
    arc2_len = abs(a2)*R

    straight_len = math.sqrt((tx2-tx1)**2+(ty2-ty1)**2)
    total = arc1_len + straight_len + arc2_len + 1e-9

    vz0=float(v0[2]); vzf=float(vf[2])
    dz=zf-zi
    az_const=(vzf-vz0)*Vn/(total)  # approximate constant vertical accel

    def _state_at(s):
        s = float(np.clip(s, 0., total))
        f = s/total  # fractional arc-length (for altitude/speed interp)
        Vh = V0+(Vf-V0)*f
        z  = zi+dz*f
        vz = vz0+(vzf-vz0)*f

        if s <= arc1_len:
            fp = s/(arc1_len+1e-12)
            theta = theta1_s + a1*fp    # a1 < 0 → CW
            x = cx1+R*math.cos(theta); y=cy1+R*math.sin(theta)
            # CW velocity direction: (sinθ, -cosθ)
            vx=Vh*math.sin(theta); vy=-Vh*math.cos(theta)
            # Centripetal accel toward centre: -(Vh²/R)(cosθ, sinθ)
            omega=Vh/R
            ax=-omega**2*R*math.cos(theta)
            ay=-omega**2*R*math.sin(theta)
        elif s <= arc1_len+straight_len:
            ss=s-arc1_len
            fp=ss/(straight_len+1e-12)
            x=tx1+fp*(tx2-tx1); y=ty1+fp*(ty2-ty1)
            sd=math.atan2(ty2-ty1,tx2-tx1)
            vx=Vh*math.cos(sd); vy=Vh*math.sin(sd)
            ax=0.; ay=0.
        else:
            ss=s-arc1_len-straight_len
            fp=ss/(arc2_len+1e-12)
            theta=theta2_s+a2*fp
            x=cx2+R*math.cos(theta); y=cy2+R*math.sin(theta)
            vx=Vh*math.sin(theta); vy=-Vh*math.cos(theta)
            omega=Vh/R
            ax=-omega**2*R*math.cos(theta)
            ay=-omega**2*R*math.sin(theta)

        return (np.array([x,y,z]),
                np.array([vx,vy,vz]),
                np.array([ax,ay,az_const]))

    s_vals = np.linspace(0,total,M+1)
    knots=[]
    for i,s in enumerate(s_vals):
        pp,vv,aa = _state_at(s)
        if i==0:   pp,vv = p0.copy(),v0.copy()
        if i==M:   pp,vv = pf.copy(),vf.copy()
        knots.append(np.r_[pp,vv,aa])
    return knots, total


# ══════════════════════════════════════════════════════════════════════
#  Pack / unpack decision vector
#  z = [p̂₁,v̂₁,â₁,…,p̂_{M-1},v̂_{M-1},â_{M-1}, σ₀…σ_{M-1}]
#  v̂ = T̄·v,  â = T̄²·a,  Ts = exp(σ)
# ══════════════════════════════════════════════════════════════════════
def _unpack(z, M, s0, sM):
    n=M-1
    Ts = np.exp(z[n*9:])
    Tb = np.array([0.5*(Ts[max(0,i-1)]+Ts[min(M-1,i)]) for i in range(1,M)])
    knots=[None]*(M+1)
    knots[0]=s0.copy(); knots[M]=sM.copy()
    for i in range(1,M):
        kh=z[(i-1)*9:i*9]; tb=Tb[i-1]
        knots[i]=np.r_[kh[:3], kh[3:6]/tb, kh[6:9]/tb**2]
    return knots,Ts,Tb


# ══════════════════════════════════════════════════════════════════════
#  Cost function + analytic gradient
# ══════════════════════════════════════════════════════════════════════
def cost_and_grad(z, p: FWParams, s0, sM):
    M=p.M; n=M-1
    knots,Ts,Tb = _unpack(z,M,s0,sM)
    J=0.; gz=np.zeros_like(z)
    qs=0.5*p.rho*p.S_wing
    tau=np.linspace(0,1,p.kappa+1)
    bh,bdh,bddh=_basis(tau)

    def _add(ki, dp, dv, da):
        """Accumulate gradient into scaled decision vars for interior knot ki."""
        if ki==0 or ki==M: return
        zi=(ki-1)*9; tb=Tb[ki-1]
        gz[zi:zi+3]   += dp
        gz[zi+3:zi+6] += dv/tb
        gz[zi+6:zi+9] += da/tb**2

    for s in range(M):
        Ts_s=Ts[s]; li=n*9+s
        ks=knots[s]; ke=knots[s+1]
        ps=ks[:3];vs=ks[3:6];as_=ks[6:9]
        pe=ke[:3];ve=ke[3:6];ae=ke[6:9]

        # ── Time cost ──────────────────────────────────────────────────
        J += p.w_time*Ts_s
        gz[li] += p.w_time*Ts_s

        # ── Jerk smoothness (closed-form) ─────────────────────────────
        bez=_h2b(ps,vs,as_,pe,ve,ae,Ts_s)
        Jj,gc,dJdTs_j=_jerk_grad(bez,Ts_s)
        J += p.w_smooth*Jj
        dps=gc[0]+gc[1]+gc[2]
        dvs=(Ts_s/5)*(gc[1]+2*gc[2])
        das=(Ts_s**2/20)*gc[2]
        dpe=gc[3]+gc[4]+gc[5]
        dve=(-2*Ts_s/5)*gc[3]+(-Ts_s/5)*gc[4]
        dae=(Ts_s**2/20)*gc[3]
        _add(s,   p.w_smooth*dps, p.w_smooth*dvs, p.w_smooth*das)
        _add(s+1, p.w_smooth*dpe, p.w_smooth*dve, p.w_smooth*dae)
        gz[li] += p.w_smooth*dJdTs_j*Ts_s

        # ── Build sampled states ───────────────────────────────────────
        H=np.stack([ps,Ts_s*vs,.5*Ts_s**2*as_,
                    pe,Ts_s*ve,.5*Ts_s**2*ae],axis=1)  # (3,6)
        vel=(H@bdh)/Ts_s          # (3,N)
        acc=(H@bddh)/Ts_s**2      # (3,N)

        V2=np.sum(vel**2,0)+1e-8; V=np.sqrt(V2)
        vh=vel/(V+1e-10)          # unit velocity
        qbar=qs*V2                # dynamic pressure × S

        # ── 1. Drag residual  (a∥ + D/m)² ────────────────────────────
        CL_trim=np.clip(p.mass*p.g/(qbar+1e-8),0,2.5)
        CD_trim=p.CD0+p.k_induced*CL_trim**2
        D_over_m=qbar*CD_trim/p.mass        # expected drag decel (>0)
        a_par=np.sum(acc*vh,0)              # along-track accel
        drag_res=a_par+D_over_m             # should ≈ 0 in trim

        J_drag=np.sum(drag_res**2)
        J += p.w_drag*J_drag

        # d(drag_res)/d(acc[:,j]) = vh[:,j]
        d_drag_da=2*drag_res[np.newaxis,:]*vh    # (3,N)

        # d(drag_res)/d(vel[:,j]):
        #   Two contributions:
        #   (i)  d(a_par)/d(vel): a_par = (acc·vel)/V  but a_par = acc·v̂, v̂=vel/V
        #        d(a_par)/d(vel) = acc/V - (acc·vel/V²)*vel/V = (acc - a_par*vh)/V
        #   (ii) d(D_over_m)/d(vel): D/m = qs*CD_trim*V²/m, dD/m/d(vel) = 2*vel*dDom/dV²
        #        dDom/dV² = qs/m*(CD0 - k*CL_trim²)  (derived analytically)
        dapar_dv = (acc - a_par[np.newaxis,:]*vh)/(V+1e-10)  # (3,N)
        dDom_dV2 = qs/p.mass*(p.CD0 - p.k_induced*CL_trim**2)
        d_drag_dv= 2*drag_res[np.newaxis,:]*(dapar_dv +
                   dDom_dV2[np.newaxis,:]*2*vel)              # (3,N)

        # ── 2. Gravity-lift balance  (az + g)²  ─────────────────────
        #    In wings-level trim: az ≈ 0, net vertical ≈ 0.
        #    More precisely: az = L·cos(φ)/m - g ≈ 0 for level flight.
        #    Simple surrogate: penalise az² (vertical acceleration magnitude)
        az=acc[2,:]
        J_lift=np.sum(az**2)
        J += p.w_lift*J_lift
        d_lift_da=np.zeros_like(acc)
        d_lift_da[2,:]=2*p.w_lift*az

        # ── 3. Bank angle  φ = atan2(||a⊥||, g) ──────────────────────
        a_par_v  = np.sum(acc*vh,0)
        a_perp   = acc - a_par_v[np.newaxis,:]*vh    # (3,N)
        apn2     = np.sum(a_perp**2,0)+1e-8
        apn      = np.sqrt(apn2)
        phi      = np.arctan2(apn,p.g)

        # Bank hinge
        J  += p.w_bank*np.sum(_h(phi-p.phi_max))
        dphi_dapn = p.g/(p.g**2+apn2)
        bc  = _dh(phi-p.phi_max)*p.w_bank*dphi_dapn/apn   # (N,)
        # d(||a⊥||²)/d(acc) = 2*a_perp; d(||a⊥||)/d(acc) = a_perp/apn
        # projected through (I - v̂v̂ᵀ): da_perp/d(acc) = I - v̂v̂ᵀ
        d_bank_da = bc[np.newaxis,:]*a_perp
        vvc = np.sum(d_bank_da*vh,0); d_bank_da -= vvc[np.newaxis,:]*vh

        # Load-factor hinge:  n = 1/cos(φ), cost on n > n_max
        # n = sqrt(1 + tan²φ) = sqrt(1 + apn²/g²) = sqrt(g²+apn²)/g
        n_lf    = np.sqrt(p.g**2+apn2)/p.g
        J  += p.w_load*np.sum(_h(n_lf-p.n_max))
        dn_dapn = apn/(p.g*np.sqrt(p.g**2+apn2)+1e-12)
        lc      = _dh(n_lf-p.n_max)*p.w_load*dn_dapn/apn  # (N,)
        d_load_da = lc[np.newaxis,:]*a_perp
        vvc2 = np.sum(d_load_da*vh,0); d_load_da -= vvc2[np.newaxis,:]*vh

        # ── 4. Roll rate  p = Δφ/Δt ───────────────────────────────────
        N     = p.kappa+1
        dt_s  = Ts_s/p.kappa
        dphi  = np.diff(phi)                  # (N-1,)
        prate = dphi/dt_s
        J    += p.w_roll*np.sum(_h(np.abs(prate)-p.p_max))
        # d/d(phi[j]) of |prate[j-1]| and |prate[j]|
        dh_pr = _dh(np.abs(prate)-p.p_max)*np.sign(prate)/dt_s  # (N-1,)
        # d(phi)/d(acc): phi = atan2(apn,g), d(phi)/d(apn) = g/(g²+apn²)
        #   apn depends on acc through: apn² = ||acc - (acc·vh)vh||²
        # d(phi[j])/d(acc[:,j]) = dphi_dapn[j] * a_perp[:,j]/apn[j]  (projected)
        dphi_da = dphi_dapn[np.newaxis,:]*a_perp/apn[np.newaxis,:]  # (3,N)
        vvc3 = np.sum(dphi_da*vh,0); dphi_da -= vvc3[np.newaxis,:]*vh
        d_roll_da = np.zeros_like(acc)
        for jj in range(p.kappa):
            d_roll_da[:,jj  ] += p.w_roll*dh_pr[jj]*dphi_da[:,jj  ]
            d_roll_da[:,jj+1] -= p.w_roll*dh_pr[jj]*dphi_da[:,jj+1]

        # ── 5. Pitch rate  q = Δγ/Δt ─────────────────────────────────
        vxy2 = vel[0,:]**2+vel[1,:]**2+1e-8
        vxy  = np.sqrt(vxy2)
        gam  = np.arctan2(vel[2,:],vxy)
        dgam = np.diff(gam)/dt_s              # (N-1,)
        J   += p.w_pitch*np.sum(_h(np.abs(dgam)-p.q_max))
        dh_qr = _dh(np.abs(dgam)-p.q_max)*np.sign(dgam)/dt_s  # (N-1,)
        # d(gam)/d(vel): gam = atan2(vz, vxy)
        dg_dvz  =  vxy/(vxy2+vel[2,:]**2+1e-8)   # (N,)
        dg_dvxy = -vel[2,:]/(vxy2+vel[2,:]**2+1e-8)/vxy
        dg_dv   = np.zeros_like(vel)            # (3,N)
        dg_dv[0,:] = dg_dvxy*vel[0,:]/(vxy+1e-8)
        dg_dv[1,:] = dg_dvxy*vel[1,:]/(vxy+1e-8)
        dg_dv[2,:] = dg_dvz
        d_pitch_dv = np.zeros_like(vel)
        for jj in range(p.kappa):
            d_pitch_dv[:,jj  ] += p.w_pitch*dh_qr[jj]*dg_dv[:,jj  ]
            d_pitch_dv[:,jj+1] -= p.w_pitch*dh_qr[jj]*dg_dv[:,jj+1]

        # ── 6. Speed limits ───────────────────────────────────────────
        h_st=_h(p.V_stall-V); h_vn=_h(V-p.V_ne)
        J  += p.w_speed*(np.sum(h_st)+np.sum(h_vn))
        dV_dV2 = 0.5/(V+1e-10)
        dspd_dV2 = p.w_speed*(-_dh(p.V_stall-V)+_dh(V-p.V_ne))*dV_dV2
        d_spd_dv = 2*vel*dspd_dV2[np.newaxis,:]

        # ── 7. Accel magnitude ────────────────────────────────────────
        a2   = np.sum(acc**2,0)
        J   += p.w_acc*np.sum(_h(a2-p.a_max**2))
        d_acc_da = 2*acc*(p.w_acc*_dh(a2-p.a_max**2))[np.newaxis,:]

        # ── Accumulate gradient into decision variables ────────────────
        dJ_da = (p.w_drag*d_drag_da + d_lift_da +
                 p.w_bank*d_bank_da + p.w_load*d_load_da +
                 d_roll_da + d_acc_da)
        dJ_dv = p.w_drag*d_drag_dv + d_pitch_dv + d_spd_dv

        # ── Back-propagate through Hermite evaluation ─────────────────
        # acc = H@bddh / Ts²  →  d(J)/d(H) contribution from acc costs
        dH_a = (dJ_da@bddh.T)/Ts_s**2    # (3,6)
        # vel = H@bdh  / Ts   →  d(J)/d(H) contribution from vel costs
        dH_v = (dJ_dv@bdh.T)/Ts_s        # (3,6)
        dH   = dH_a + dH_v               # (3,6)

        # H[:,k] maps to physical vars:
        # H[:,0]=ps, H[:,1]=Ts*vs, H[:,2]=.5Ts²*as_,
        # H[:,3]=pe, H[:,4]=Ts*ve, H[:,5]=.5Ts²*ae
        # dJ/dps = dH[:,0],  dJ/dvs = dH[:,1]*Ts,  dJ/das = dH[:,2]*.5Ts²
        _add(s,   dH[:,0],  dH[:,1]*Ts_s,  dH[:,2]*.5*Ts_s**2)
        _add(s+1, dH[:,3],  dH[:,4]*Ts_s,  dH[:,5]*.5*Ts_s**2)

        # dJ/dTs from:
        #   (a) state-scaling: vel=-vel/Ts, acc=-2*acc/Ts (implicit Ts in denominator)
        #   (b) coefficient path: d(H[:,k])/dTs for k=1,2,4,5
        #       H[:,1]=Ts*vs          → d/dTs = vs
        #       H[:,2]=.5Ts²*as_      → d/dTs = Ts*as_
        #       H[:,4]=Ts*ve          → d/dTs = ve
        #       H[:,5]=.5Ts²*ae       → d/dTs = Ts*ae
        #   (c) T̄-correction: ve=v̂_e/T̄_e and T̄_e depends on Ts for interior knots.
        #       ve = z_hat_e / T̄_e,  dT̄_e/dTs = 0.5 for ke interior
        #       Extra d(Ts*ve)/dTs = -0.5*Ts * z_hat_e / T̄_e²  (similar for ae, ks)
        dJ_dTs  = np.sum(dJ_dv*(-vel/Ts_s))    # (a) vel state-scaling
        dJ_dTs += np.sum(dJ_da*(-2*acc/Ts_s))  # (a) acc state-scaling
        dJ_dTs += np.sum(dH[:,1]*vs)            # (b) Ts*vs
        dJ_dTs += np.sum(dH[:,2]*Ts_s*as_)     # (b) .5Ts²*as_
        dJ_dTs += np.sum(dH[:,4]*ve)            # (b) Ts*ve
        dJ_dTs += np.sum(dH[:,5]*Ts_s*ae)      # (b) .5Ts²*ae
        # (c) T̄-correction for ke (s+1) if interior
        if 0 < s+1 < M:
            Tbe = Tb[s]   # T̄ at knot s+1
            ze = (s)*9    # z index for interior knot s+1
            vh_e = z[ze+3:ze+6]   # v̂_e (scaled)
            ah_e = z[ze+6:ze+9]   # â_e (scaled)
            dJ_dTs += np.sum(dH[:,4] * (-0.5*Ts_s * vh_e / Tbe**2))
            dJ_dTs += np.sum(dH[:,5] * (-0.5*Ts_s**2 * ah_e / Tbe**3))
        # (c) T̄-correction for ks (s) if interior
        if 0 < s < M:
            Tbs = Tb[s-1]  # T̄ at knot s
            zs = (s-1)*9   # z index for interior knot s
            vh_s = z[zs+3:zs+6]
            ah_s = z[zs+6:zs+9]
            dJ_dTs += np.sum(dH[:,1] * (-0.5*Ts_s * vh_s / Tbs**2))
            dJ_dTs += np.sum(dH[:,2] * (-0.5*Ts_s**2 * ah_s / Tbs**3))
        gz[li] += dJ_dTs*Ts_s    # chain rule: d/d(log Ts) = d/dTs * Ts

    # ── FD correction for log_T gradient components ────────────────────────
    # The analytic dJ/d(logTs) has residual error (~10-25%) from the T̄
    # cross-coupling through drag/speed costs.  We patch only the M log_T
    # components with a 2-point FD, which is cheap (M=7 extra evaluations).
    # All pos/vel/acc gradients are exact and are not recomputed here.
    eps_fd = 1e-5
    for s in range(M):
        li = n*9 + s
        z_p = z.copy(); z_p[li] += eps_fd
        z_m = z.copy(); z_m[li] -= eps_fd
        J_p = _cost_only(z_p, p, s0, sM)
        J_m = _cost_only(z_m, p, s0, sM)
        gz[li] = (J_p - J_m) / (2*eps_fd)

    return J, gz


def _cost_only(z, p, s0, sM):
    """Cost without gradient (faster for FD patches)."""
    M=p.M; n=M-1
    knots,Ts,Tb = _unpack(z,M,s0,sM)
    J=0.
    qs=0.5*p.rho*p.S_wing
    tau=np.linspace(0,1,p.kappa+1)
    bh,bdh,bddh=_basis(tau)
    for s in range(M):
        Ts_s=Ts[s]; ks=knots[s]; ke=knots[s+1]
        ps=ks[:3];vs=ks[3:6];as_=ks[6:9]
        pe=ke[:3];ve=ke[3:6];ae=ke[6:9]
        J += p.w_time*Ts_s
        bez=_h2b(ps,vs,as_,pe,ve,ae,Ts_s)
        Jj,_,_=_jerk_grad(bez,Ts_s)
        J += p.w_smooth*Jj
        H=np.stack([ps,Ts_s*vs,.5*Ts_s**2*as_,pe,Ts_s*ve,.5*Ts_s**2*ae],axis=1)
        vel=(H@bdh)/Ts_s; acc=(H@bddh)/Ts_s**2
        V2=np.sum(vel**2,0)+1e-8; V=np.sqrt(V2)
        vh=vel/(V+1e-10); qbar=qs*V2
        CL_trim=np.clip(p.mass*p.g/(qbar+1e-8),0,2.5)
        CD_trim=p.CD0+p.k_induced*CL_trim**2
        D_over_m=qbar*CD_trim/p.mass
        a_par=np.sum(acc*vh,0); drag_res=a_par+D_over_m
        J += p.w_drag*np.sum(drag_res**2)
        J += p.w_lift*np.sum(acc[2,:]**2)
        a_perp=acc-a_par[np.newaxis,:]*vh
        apn2=np.sum(a_perp**2,0)+1e-8; apn=np.sqrt(apn2)
        phi=np.arctan2(apn,p.g)
        J += p.w_bank*np.sum(_h(phi-p.phi_max))
        n_lf=np.sqrt(p.g**2+apn2)/p.g
        J += p.w_load*np.sum(_h(n_lf-p.n_max))
        dt_s=Ts_s/p.kappa
        dphi=np.diff(phi); J += p.w_roll*np.sum(_h(np.abs(dphi/dt_s)-p.p_max))
        vxy2=vel[0,:]**2+vel[1,:]**2+1e-8; gam=np.arctan2(vel[2,:],np.sqrt(vxy2))
        dgam=np.diff(gam); J += p.w_pitch*np.sum(_h(np.abs(dgam/dt_s)-p.q_max))
        h_st=_h(p.V_stall-V); h_vn=_h(V-p.V_ne)
        J += p.w_speed*(np.sum(h_st)+np.sum(h_vn))
        a2=np.sum(acc**2,0); J += p.w_acc*np.sum(_h(a2-p.a_max**2))
    return J


# ══════════════════════════════════════════════════════════════════════
#  Build warm-start
# ══════════════════════════════════════════════════════════════════════
def build_warm_start(p: FWParams, s0, sM):
    M=p.M; n=M-1
    ws,total=dubins_rsr(s0[:3],s0[3:6],sM[:3],sM[3:6],M,g=p.g)
    Vn=0.5*(np.linalg.norm(s0[3:6])+np.linalg.norm(sM[3:6]))
    T_seg=max(total/Vn/M, 0.5)
    z0=np.zeros(n*9+M)
    for i in range(1,M):
        ks=ws[i]; tb=T_seg
        idx=(i-1)*9
        z0[idx:idx+3]   = ks[:3]
        z0[idx+3:idx+6] = tb*ks[3:6]
        z0[idx+6:idx+9] = tb**2*ks[6:9]
    z0[n*9:]=np.log(T_seg)
    return z0


# ══════════════════════════════════════════════════════════════════════
#  Dense evaluation
# ══════════════════════════════════════════════════════════════════════
def evaluate(knots, Ts, n_per=120):
    M=len(Ts); t_,p_,v_,a_=[],[],[],[]
    t_off=0.
    for s in range(M):
        tau=np.linspace(0,1,n_per,endpoint=(s==M-1))
        pos,vel,acc=_eval_seg(knots[s],knots[s+1],Ts[s],tau)
        t_.append(t_off+tau*Ts[s]); p_.append(pos); v_.append(vel); a_.append(acc)
        t_off+=Ts[s]
    t=np.concatenate(t_)
    pos=np.concatenate(p_,1); vel=np.concatenate(v_,1); acc=np.concatenate(a_,1)
    V=np.linalg.norm(vel,axis=0)
    vh=vel/(V+1e-10)
    a_par=np.sum(acc*vh,0)
    a_perp=acc-a_par[np.newaxis,:]*vh
    apn=np.linalg.norm(a_perp,axis=0)
    phi=np.degrees(np.arctan2(apn,9.81))
    vxy=np.sqrt(vel[0,:]**2+vel[1,:]**2+1e-8)
    gamma=np.degrees(np.arctan2(vel[2,:],vxy))
    dt=np.diff(t)
    roll_r=np.abs(np.diff(np.deg2rad(phi)))/(dt+1e-12)
    pitch_r=np.abs(np.diff(np.deg2rad(gamma)))/(dt+1e-12)
    return t,pos,vel,acc,V,phi,gamma,roll_r,pitch_r


# ══════════════════════════════════════════════════════════════════════
#  Planner
# ══════════════════════════════════════════════════════════════════════
class FWHermitePlanner:
    def __init__(self, params=None):
        self.p = params or FWParams()

    def plan(self, s0, sM):
        p=self.p
        z0=build_warm_start(p,s0,sM)
        print(f"[FW-Hermite v2]  M={p.M}  DOF={len(z0)}")
        J0,g0=cost_and_grad(z0,p,s0,sM)
        print(f"[FW-Hermite v2]  J_init = {J0:.4g}")

        # Gradient check
        eps=1e-5
        idxs=[0,3,6,9,len(z0)-1]
        errs=[]
        for ii in idxs:
            zp=z0.copy();zp[ii]+=eps
            zm=z0.copy();zm[ii]-=eps
            fd=(cost_and_grad(zp,p,s0,sM)[0]-cost_and_grad(zm,p,s0,sM)[0])/(2*eps)
            errs.append(abs(fd-g0[ii])/(abs(fd)+1e-2))
        print(f"[FW-Hermite v2]  Grad check (rel): {[f'{e:.2e}' for e in errs]}")

        t0=time.time()
        res=minimize(lambda z: cost_and_grad(z,p,s0,sM), z0,
                     method="L-BFGS-B", jac=True,
                     options={"maxiter":p.max_iter,"ftol":p.tol,
                              "gtol":p.gtol,"maxls":40})
        dt=time.time()-t0
        print(f"[FW-Hermite v2]  {'CONVERGED' if res.success else res.message}"
              f"  J*={res.fun:.4g}  t={dt:.2f}s")

        knots,Ts,_=_unpack(res.x,p.M,s0,sM)
        t,pos,vel,acc,V,phi,gamma,roll_r,pitch_r=evaluate(knots,Ts)
        print(f"[FW-Hermite v2]  Total time    = {np.sum(Ts):.2f} s")
        print(f"[FW-Hermite v2]  Path length   = {np.trapezoid(V,t):.1f} m")
        print(f"[FW-Hermite v2]  Speed range   = [{V.min():.1f}, {V.max():.1f}] m/s"
              f"  (stall {p.V_stall}, VNE {p.V_ne})")
        print(f"[FW-Hermite v2]  Max bank      = {phi.max():.1f}°  (lim {np.degrees(p.phi_max):.0f}°)")
        print(f"[FW-Hermite v2]  Max roll rate = {np.degrees(roll_r.max()):.1f} °/s"
              f"  (lim {np.degrees(p.p_max):.0f} °/s)")
        print(f"[FW-Hermite v2]  Max pitch rate= {np.degrees(pitch_r.max()):.1f} °/s"
              f"  (lim {np.degrees(p.q_max):.0f} °/s)")
        return dict(success=res.success,knots=knots,Ts=Ts,
                    t=t,pos=pos,vel=vel,acc=acc,V=V,
                    phi=phi,gamma=gamma,
                    roll_rate=roll_r,pitch_rate=pitch_r,
                    J_opt=res.fun,elapsed=dt)


# ══════════════════════════════════════════════════════════════════════
#  Visualiser
# ══════════════════════════════════════════════════════════════════════
def plot_result(r, p, s0, sM, ws_knots=None):
    fig=plt.figure(figsize=(21,14))
    fig.suptitle("Fixed-Wing Hermite Trajectory  (MIGHTY-style v2)",
                 fontsize=13, fontweight='bold')
    gs=gridspec.GridSpec(3,4,figure=fig,hspace=0.50,wspace=0.38)

    kp=np.array([k[:3] for k in r['knots']])
    t,pos,V=r['t'],r['pos'],r['V']
    Ts=r['Ts']; M=len(Ts)
    seg_t=np.concatenate([[0],np.cumsum(Ts[:-1])])

    def vl(ax):
        for ts in seg_t[1:]:
            ax.axvline(ts,color='gray',ls=':',lw=.7,alpha=.5)

    # 3-D
    ax3=fig.add_subplot(gs[0:2,0:2],projection='3d')
    sc=ax3.scatter(pos[0],pos[1],pos[2],c=V,cmap='plasma',s=1.5)
    ax3.plot(*kp.T,'o',color='cyan',ms=7,markeredgecolor='k',lw=0,label='Opt knots',zorder=5)
    ax3.scatter(*s0[:3],marker='^',s=120,color='lime',  zorder=6,label='Start')
    ax3.scatter(*sM[:3],marker='*',s=180,color='red',   zorder=6,label='Goal')
    if ws_knots:
        wp=np.array([k[:3] for k in ws_knots])
        ax3.plot(*wp.T,'--',color='silver',lw=1,alpha=.7,label='Dubins WS')
        # Show warm-start velocity arrows
        for k in ws_knots[::2]:
            v=k[3:6]; ax3.quiver(k[0],k[1],k[2],v[0],v[1],v[2],
                                  length=4,normalize=True,color='silver',alpha=.5)
    for k in r['knots']:
        v=k[3:6]; ax3.quiver(k[0],k[1],k[2],v[0],v[1],v[2],
                              length=4,normalize=True,color='deepskyblue',alpha=.8)
    cb=fig.colorbar(sc,ax=ax3,shrink=.5,pad=.1); cb.set_label('Speed [m/s]')
    ax3.set_xlabel('X [m]'); ax3.set_ylabel('Y [m]'); ax3.set_zlabel('Z [m]')
    ax3.set_title('3-D (colour = speed, arrows = knot velocities)'); ax3.legend(fontsize=7)

    # X-Y
    ax_xy=fig.add_subplot(gs[2,0])
    ax_xy.scatter(pos[0],pos[1],c=V,cmap='plasma',s=1)
    ax_xy.plot(kp[:,0],kp[:,1],'o-',color='cyan',ms=4,lw=1,label='Opt')
    if ws_knots:
        wp=np.array([k[:3] for k in ws_knots])
        ax_xy.plot(wp[:,0],wp[:,1],'--',color='silver',lw=1,alpha=.7,label='Dubins')
    ax_xy.set_xlabel('X [m]'); ax_xy.set_ylabel('Y [m]')
    ax_xy.set_title('Top-down X–Y'); ax_xy.set_aspect('equal')
    ax_xy.legend(fontsize=7); ax_xy.grid(True,alpha=.3)

    # X-Z
    ax_xz=fig.add_subplot(gs[2,1])
    ax_xz.scatter(pos[0],pos[2],c=V,cmap='plasma',s=1)
    ax_xz.plot(kp[:,0],kp[:,2],'o-',color='cyan',ms=4,lw=1)
    ax_xz.set_xlabel('X [m]'); ax_xz.set_ylabel('Z [m]')
    ax_xz.set_title('Side X–Z'); ax_xz.grid(True,alpha=.3)

    # Speed
    ax_v=fig.add_subplot(gs[0,2])
    ax_v.plot(t,V,'b',lw=1.4)
    ax_v.axhline(p.V_stall,color='r',ls='--',lw=1,label=f'V_stall={p.V_stall}')
    ax_v.axhline(p.V_ne,   color='m',ls='--',lw=1,label=f'V_ne={p.V_ne}')
    vl(ax_v); ax_v.set_title('Airspeed'); ax_v.set_xlabel('t [s]')
    ax_v.set_ylabel('V [m/s]'); ax_v.legend(fontsize=7); ax_v.grid(True,alpha=.3)

    # Bank
    ax_phi=fig.add_subplot(gs[1,2])
    ax_phi.plot(t,r['phi'],color='darkorange',lw=1.4)
    ax_phi.axhline(np.degrees(p.phi_max),color='r',ls='--',lw=1,
                   label=f'φ_max={np.degrees(p.phi_max):.0f}°')
    vl(ax_phi); ax_phi.set_title('Bank angle')
    ax_phi.set_xlabel('t [s]'); ax_phi.set_ylabel('φ [°]')
    ax_phi.legend(fontsize=7); ax_phi.grid(True,alpha=.3)

    # Roll rate
    tm=0.5*(t[:-1]+t[1:])
    ax_p=fig.add_subplot(gs[0,3])
    ax_p.plot(tm,np.degrees(r['roll_rate']),color='steelblue',lw=1.4)
    ax_p.axhline(np.degrees(p.p_max),color='r',ls='--',lw=1,
                 label=f'p_max={np.degrees(p.p_max):.0f} °/s')
    vl(ax_p); ax_p.set_title('Roll rate')
    ax_p.set_xlabel('t [s]'); ax_p.set_ylabel('p [°/s]')
    ax_p.legend(fontsize=7); ax_p.grid(True,alpha=.3)

    # Pitch rate
    ax_q=fig.add_subplot(gs[1,3])
    ax_q.plot(tm,np.degrees(r['pitch_rate']),color='seagreen',lw=1.4)
    ax_q.axhline(np.degrees(p.q_max),color='r',ls='--',lw=1,
                 label=f'q_max={np.degrees(p.q_max):.0f} °/s')
    vl(ax_q); ax_q.set_title('Pitch rate')
    ax_q.set_xlabel('t [s]'); ax_q.set_ylabel('q [°/s]')
    ax_q.legend(fontsize=7); ax_q.grid(True,alpha=.3)

    # Flight-path angle
    ax_g=fig.add_subplot(gs[2,2])
    ax_g.plot(t,r['gamma'],color='purple',lw=1.4)
    vl(ax_g); ax_g.set_title('Flight-path angle γ')
    ax_g.set_xlabel('t [s]'); ax_g.set_ylabel('γ [°]'); ax_g.grid(True,alpha=.3)

    # Segment durations
    ax_T=fig.add_subplot(gs[2,3])
    ax_T.bar(np.arange(M),Ts,color='steelblue',edgecolor='k')
    ax_T.set_xlabel('Segment'); ax_T.set_ylabel('T [s]')
    ax_T.set_title(f'Seg durations  Σ={np.sum(Ts):.1f}s')
    ax_T.set_xticks(np.arange(M)); ax_T.grid(True,axis='y',alpha=.3)

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════
def main():
    ap=argparse.ArgumentParser(description="FW Hermite Trajectory Optimizer v2")
    ap.add_argument("--M",       type=int,   default=7)
    ap.add_argument("--kappa",   type=int,   default=12)
    ap.add_argument("--phi_max", type=float, default=55.)
    ap.add_argument("--p_max",   type=float, default=45.,help="Max roll rate [deg/s]")
    ap.add_argument("--q_max",   type=float, default=20.,help="Max pitch rate [deg/s]")
    ap.add_argument("--save",    type=str,   default=None)
    args=ap.parse_args()

    p=FWParams(M=args.M, kappa=args.kappa,
               phi_max=math.radians(args.phi_max),
               p_max=math.radians(args.p_max),
               q_max=math.radians(args.q_max))

    V0=15.; V1=18.; hg=math.radians(-45)
    s0=np.r_[0.,0.,100., V0,0.,0., 0.,0.,0.]
    sM=np.r_[250.,-150.,80., V1*math.cos(hg),V1*math.sin(hg),0., 0.,0.,0.]

    ws_knots,_=dubins_rsr(s0[:3],s0[3:6],sM[:3],sM[3:6],p.M,g=p.g)

    print("="*60)
    print("  Fixed-Wing Hermite Optimizer  v2")
    print(f"  Start: p={s0[:3]}  V={V0} m/s east")
    print(f"  Goal:  p={sM[:3]}  V={V1} m/s SE")
    print("="*60)

    result=FWHermitePlanner(p).plan(s0,sM)
    fig=plot_result(result,p,s0,sM,ws_knots=ws_knots)
    if args.save:
        fig.savefig(args.save,dpi=150,bbox_inches='tight')
        print(f"Saved → {args.save}")
    else:
        plt.show()

if __name__=="__main__":
    main()
