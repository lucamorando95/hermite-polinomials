#!/usr/bin/env python3
"""
fw_hermite_opt.py
=================
Fixed-Wing Hermite Spline Trajectory Optimizer  —  MIGHTY-style
----------------------------------------------------------------
Python reference implementation matching fw_hermite_planner.cpp.

Optimises a quintic Hermite spline for a fixed-wing UAV from
waypoint A to waypoint B with prescribed boundary velocities and
accelerations, subject to soft-constrained:
  - Lift / drag aerodynamic residual
  - Bank angle limit  (coordinated turn model)
  - Speed limits (stall + VNE)
  - Acceleration magnitude limit
  - Integrated jerk smoothness (closed-form)
  - Mission time

Warm-start: Dubins RSR path (2-D), altitude linearly interpolated.

Optimiser: scipy L-BFGS-B (matches MIGHTY's L-BFGS approach).
           AD-free: all gradients computed analytically via the
           Hermite→Bézier chain-rule equations in the paper.

Usage
-----
  python3 fw_hermite_opt.py          # runs built-in demo scenario
  python3 fw_hermite_opt.py --help
"""

import argparse
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ═════════════════════════════════════════════════════════════════════════════
#  Parameters
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class FWParams:
    # Spline
    M: int   = 6      # segments
    kappa: int = 10   # samples per segment for integral costs

    # Weights
    w_time:   float = 5e1
    w_smooth: float = 1e-1
    w_aero:   float = 5e1
    w_bank:   float = 1e3
    w_speed:  float = 1e3
    w_acc:    float = 5e2

    # Fixed-wing aerodynamics (point-mass)
    mass:      float = 2.5     # kg
    g:         float = 9.81    # m/s²
    rho:       float = 1.225   # kg/m³
    S_wing:    float = 0.35    # m²
    CL0:       float = 0.3
    CLa:       float = 5.0
    CD0:       float = 0.025
    k_induced: float = 0.045
    V_stall:   float = 8.0     # m/s
    V_ne:      float = 30.0    # m/s

    # Limits
    phi_max:   float = math.radians(55)  # rad
    a_max:     float = 15.0              # m/s²

    # Optimiser
    max_iter:  int   = 800
    tol:       float = 1e-5
    gtol:      float = 1e-5


# ═════════════════════════════════════════════════════════════════════════════
#  Quintic Hermite basis  (scalar τ ∈ [0,1])
# ═════════════════════════════════════════════════════════════════════════════
def _quintic_basis(tau: np.ndarray):
    """Returns h (6,N), dh (6,N), ddh (6,N) for array of τ values."""
    t = np.asarray(tau, dtype=float)
    t2, t3, t4, t5 = t**2, t**3, t**4, t**5
    h = np.array([
        1 - 10*t3 + 15*t4 -  6*t5,
        t  -  6*t3 +  8*t4 -  3*t5,
        0.5*t2 - 1.5*t3 + 1.5*t4 - 0.5*t5,
        10*t3 - 15*t4 +  6*t5,
        -4*t3 +  7*t4 -  3*t5,
        0.5*t3 -    t4 + 0.5*t5,
    ])
    dh = np.array([
        -30*t2 + 60*t3 - 30*t4,
         1 - 18*t2 + 32*t3 - 15*t4,
         t  -  4.5*t2 +  6*t3 - 2.5*t4,
         30*t2 - 60*t3 + 30*t4,
        -12*t2 + 28*t3 - 15*t4,
          1.5*t2 -  4*t3 + 2.5*t4,
    ])
    ddh = np.array([
        -60*t + 180*t2 - 120*t3,
        -36*t +  96*t2 -  60*t3,
          1 -   9*t  +  18*t2 -  10*t3,
         60*t - 180*t2 + 120*t3,
        -24*t +  84*t2 -  60*t3,
           3*t -  12*t2 +  10*t3,
    ])
    return h, dh, ddh


def _eval_segment(ks_p, ks_v, ks_a, ke_p, ke_v, ke_a, Ts, tau_arr):
    """
    Evaluate position, velocity, acceleration on segment s at τ values.
    Returns pos (3,N), vel (3,N), acc (3,N).
    The Hermite parameter encoding is:
      H = [p_s, Ts*v_s, 0.5*Ts²*a_s, p_e, Ts*v_e, 0.5*Ts²*a_e]
    """
    H = np.stack([ks_p, Ts*ks_v, 0.5*Ts**2*ks_a,
                  ke_p, Ts*ke_v, 0.5*Ts**2*ke_a], axis=1)  # (3,6)
    h, dh, ddh = _quintic_basis(tau_arr)                    # (6,N)
    pos =  H @ h                                            # (3,N)
    vel = (H @ dh)  / Ts
    acc = (H @ ddh) / Ts**2
    return pos, vel, acc


# ═════════════════════════════════════════════════════════════════════════════
#  Hermite → Bézier  (MIGHTY eq. 2)
# ═════════════════════════════════════════════════════════════════════════════
def _hermite_to_bezier(ps, vs, as_, pe, ve, ae, Ts):
    """Returns control points c[0..5] each shape (3,)."""
    c = [None]*6
    c[0] = ps.copy()
    c[1] = ps + (Ts/5)*vs
    c[2] = ps + (2*Ts/5)*vs + (Ts**2/20)*as_
    c[3] = pe - (2*Ts/5)*ve + (Ts**2/20)*ae
    c[4] = pe - (Ts/5)*ve
    c[5] = pe.copy()
    return c


# ═════════════════════════════════════════════════════════════════════════════
#  Closed-form jerk integral  (MIGHTY eq. 6, diagonal surrogate)
# ═════════════════════════════════════════════════════════════════════════════
_G_DIAG = np.array([1/5, 2/15, 1/5])   # diagonal G weights

def _jerk_integral(c, Ts):
    """Jerk integral for one Bézier segment. c: list of 6 (3,) arrays."""
    delta = [c[m+3] - 3*c[m+2] + 3*c[m+1] - c[m] for m in range(3)]
    Cs = 3600.0 * Ts**(-5)
    J = Cs * sum(_G_DIAG[m] * float(delta[m] @ delta[m]) for m in range(3))
    return J


def _jerk_integral_grad(c, Ts):
    """
    Returns (J, dJ/dc[0..5], dJ/dTs) for one Bézier segment.
    Uses MIGHTY eq. (3)-(5) chain rule.
    """
    delta = [c[m+3] - 3*c[m+2] + 3*c[m+1] - c[m] for m in range(3)]
    Cs = 3600.0 * Ts**(-5)

    # Cost
    J = Cs * sum(_G_DIAG[m] * float(delta[m] @ delta[m]) for m in range(3))

    # dJ/dΔm = 2*Cs*G_diag[m]*Δm
    dJ_ddelta = [2*Cs*_G_DIAG[m]*delta[m] for m in range(3)]

    # dJ/dc[j] = Σ_m α_{j,m} * dJ/dΔm   (alpha coeffs: -1,3,-3,1 at j=m..m+3)
    alpha = [-1, 3, -3, 1]
    dJ_dc = [np.zeros(3) for _ in range(6)]
    for m in range(3):
        for k, j in enumerate(range(m, m+4)):
            dJ_dc[j] += alpha[k] * dJ_ddelta[m]

    # dJ/dTs: explicit term (−5/Ts)*J + coefficient path
    # From MIGHTY eq. (4)+(5) the ∂cs/∂Ts terms:
    # ∇Ts c0 = 0,  c1 = (1/5)vs,  c2 = (2/5)vs + (Ts/10)as,
    # c3 = -(2/5)ve + (Ts/10)ae,  c4 = -(1/5)ve,  c5 = 0
    # We need vs, as, ve, ae which are encoded in c via the map.
    # Recover: vs = (c1-c0)*5/Ts,  as = (c2-c0-(2Ts/5)vs)*20/Ts²,
    #          ve = (c5-c4)*5/Ts,  ae = (c3-c5+(2Ts/5)ve)*20/Ts²
    vs = (c[1] - c[0]) * 5.0 / Ts
    ve = (c[5] - c[4]) * 5.0 / Ts
    as_ = (c[2] - c[0] - (2*Ts/5)*vs) * 20.0 / Ts**2
    ae  = (c[3] - c[5] + (2*Ts/5)*ve) * 20.0 / Ts**2
    dcs_dTs = [
        np.zeros(3),
        vs / 5.0,
        (2.0/5)*vs + (Ts/10)*as_,
        -(2.0/5)*ve + (Ts/10)*ae,
        -ve / 5.0,
        np.zeros(3),
    ]
    coeff_path = sum(float(dJ_dc[k] @ dcs_dTs[k]) for k in range(6))
    dJ_dTs = (-5.0/Ts)*J + coeff_path

    return J, dJ_dc, dJ_dTs


# ═════════════════════════════════════════════════════════════════════════════
#  Smooth hinge  ϕ(x) = (1/μ) log(1 + exp(μ·x))
# ═════════════════════════════════════════════════════════════════════════════
_MU = 50.0

def _hinge(x):
    # Numerically stable version
    mx = _MU * x
    # log(1+exp(mx)) = mx + log(1+exp(-mx)) for mx>0
    return np.where(mx > 0,
                    (mx + np.log1p(np.exp(-mx))) / _MU,
                    np.log1p(np.exp(mx)) / _MU)

def _hinge_grad(x):
    mx = _MU * x
    return np.where(mx > 0,
                    1.0 / (1.0 + np.exp(-mx)),
                    np.exp(mx) / (1.0 + np.exp(mx)))


# ═════════════════════════════════════════════════════════════════════════════
#  Dubins warm-start  (RSR in 2-D, linear altitude interpolation)
# ═════════════════════════════════════════════════════════════════════════════
def dubins_warmstart(p_start, v_start, p_goal, v_goal, M, R=20.0):
    """
    Returns (M+1) 3-D waypoints along a Dubins RSR path.
    p: (3,), v: (3,) — 3-D position and velocity vectors.
    """
    xi, yi, zi = p_start
    xf, yf, zf = p_goal
    hi = math.atan2(v_start[1], v_start[0])
    hf = math.atan2(v_goal[1],  v_goal[0])

    # Right-turn circle centres
    cx1 = xi + R * math.sin(hi);  cy1 = yi - R * math.cos(hi)
    cx2 = xf + R * math.sin(hf);  cy2 = yf - R * math.cos(hf)

    dx = cx2 - cx1;  dy = cy2 - cy1
    beta = math.atan2(dy, dx)

    # Tangent point on first circle
    tx1 = cx1 + R * math.cos(beta - math.pi/2)
    ty1 = cy1 + R * math.sin(beta - math.pi/2)

    pts = []
    for i in range(M + 1):
        frac = i / M
        x = xi + frac * (xf - xi)
        y = yi + frac * (yf - yi)
        # Add a lateral bow toward tangent point (smooth Dubins approximation)
        bump = 4.0 * frac * (1.0 - frac)
        x += bump * 0.4 * (tx1 - 0.5*(xi+xf))
        y += bump * 0.4 * (ty1 - 0.5*(yi+yf))
        z  = zi + frac * (zf - zi)
        pts.append(np.array([x, y, z]))

    pts[0]  = p_start.copy()
    pts[-1] = p_goal.copy()
    return pts


# ═════════════════════════════════════════════════════════════════════════════
#  Pack / unpack decision vector
#  z = [p̂₁,v̂₁,â₁, …, p̂_{M-1},v̂_{M-1},â_{M-1}, σ₀,…,σ_{M-1}]
#  where v̂ᵢ = T̄ᵢ·vᵢ,  âᵢ = T̄ᵢ²·aᵢ,  σs = log(Ts)
# ═════════════════════════════════════════════════════════════════════════════
def _pack(knots_pva: List[np.ndarray],   # (M-1) × [p,v,a] each (9,)
          log_T: np.ndarray) -> np.ndarray:
    return np.concatenate([np.concatenate(knots_pva), log_T])

def _unpack(z, M, start_state, goal_state, T_bars=None):
    """
    Returns:
      knots: list of (M+1) arrays, each [p(3), v(3), a(3)] = shape (9,)
      Ts:    (M,) array of segment durations
    """
    n_int = M - 1
    log_T = z[n_int*9:]
    Ts = np.exp(log_T)

    # Interior scaled knots
    knots_hat = [z[i*9:(i+1)*9] for i in range(n_int)]

    # Build T̄ᵢ for each interior knot i=1..M-1
    T_bar_arr = np.array([
        0.5*(Ts[max(0,i-1)] + Ts[min(M-1,i)]) if 0 < i < M
        else (Ts[0] if i == 0 else Ts[M-1])
        for i in range(1, M)
    ])

    knots = [None] * (M + 1)
    knots[0] = start_state.copy()
    knots[M] = goal_state.copy()

    for i in range(1, M):
        kh = knots_hat[i-1]   # scaled [p̂, v̂, â]
        Tb = T_bar_arr[i-1]
        p_ = kh[0:3]
        v_ = kh[3:6] / Tb
        a_ = kh[6:9] / Tb**2
        knots[i] = np.concatenate([p_, v_, a_])

    return knots, Ts, T_bar_arr


# ═════════════════════════════════════════════════════════════════════════════
#  Cost function and gradient
# ═════════════════════════════════════════════════════════════════════════════
def cost_and_grad(z: np.ndarray, p: FWParams,
                  start_state: np.ndarray, goal_state: np.ndarray):
    """
    Returns (J, dJ/dz).
    start_state, goal_state: (9,) arrays [p, v, a].
    """
    M = p.M
    n_int = M - 1
    knots, Ts, T_bars = _unpack(z, M, start_state, goal_state)

    J_total = 0.0
    grad_z  = np.zeros_like(z)
    # dJ/d(scaled knot i) is grad_z[i*9:(i+1)*9] for i=0..n_int-1
    # dJ/d(log_Ts[s])     is grad_z[n_int*9 + s]

    q_scale = 0.5 * p.rho * p.S_wing

    # ── Sample points in τ ───────────────────────────────────────────────────
    tau_arr = np.linspace(0, 1, p.kappa + 1)   # (κ+1,)
    h, dh, ddh = _quintic_basis(tau_arr)         # (6, κ+1)

    for s in range(M):
        Ts_s = Ts[s]
        log_Ts_idx = n_int*9 + s

        ks = knots[s];   ke = knots[s+1]
        ps = ks[0:3];    vs = ks[3:6];    as_ = ks[6:9]
        pe = ke[0:3];    ve = ke[3:6];    ae  = ke[6:9]

        # ── Hermite→Bézier and jerk cost ─────────────────────────────────────
        bez = _hermite_to_bezier(ps, vs, as_, pe, ve, ae, Ts_s)
        J_jerk, dJ_dc, dJ_dTs_jerk = _jerk_integral_grad(bez, Ts_s)
        J_total += p.w_smooth * J_jerk

        # ── Time cost ────────────────────────────────────────────────────────
        J_total  += p.w_time * Ts_s
        grad_z[log_Ts_idx] += p.w_time * Ts_s   # d(Ts)/d(log_Ts) = Ts

        # Accumulate jerk gradient into decision-variable gradient
        # Chain rule:  dJ/dz = dJ/dc · dc/dH · dH/dz
        # For interior knots: H params are [p, T·v, 0.5T²·a]
        # For boundary knots they are fixed.
        def _add_grad_to_knot(knot_idx, dp, dv, da):
            """Accumulate position/velocity/accel gradients into grad_z."""
            if knot_idx == 0 or knot_idx == M:
                return  # boundary — fixed
            zi = (knot_idx - 1) * 9
            Tb = T_bars[knot_idx - 1]
            # dJ/d(p̂) = dJ/dp   (p̂ = p, no scaling)
            grad_z[zi:zi+3]   += p.w_smooth * dp
            # dJ/d(v̂) = dJ/dv / T̄   (v = v̂/T̄)
            grad_z[zi+3:zi+6] += p.w_smooth * dv / Tb
            # dJ/d(â) = dJ/da / T̄²
            grad_z[zi+6:zi+9] += p.w_smooth * da / Tb**2

        # Propagate Bézier control-point gradients back to Hermite vars
        # dc/dH:  c[0]=ps, c[1]=ps+(Ts/5)vs, c[2]=ps+(2Ts/5)vs+(Ts²/20)as
        #         c[3]=pe-(2Ts/5)ve+(Ts²/20)ae, c[4]=pe-(Ts/5)ve, c[5]=pe
        # Therefore:
        # dJ/dps = dJ/dc0 + dJ/dc1 + dJ/dc2
        # dJ/dvs = (Ts/5)*(dJ/dc1 + 2*dJ/dc2)
        # dJ/das = (Ts²/20)*dJ/dc2
        # Similarly for pe, ve, ae
        dJ_dps = dJ_dc[0] + dJ_dc[1] + dJ_dc[2]
        dJ_dvs = (Ts_s/5)*(dJ_dc[1] + 2*dJ_dc[2])
        dJ_das = (Ts_s**2/20)*dJ_dc[2]
        dJ_dpe = dJ_dc[3] + dJ_dc[4] + dJ_dc[5]
        dJ_dve = (-2*Ts_s/5)*dJ_dc[3] + (-Ts_s/5)*dJ_dc[4]
        dJ_dae = (Ts_s**2/20)*dJ_dc[3]

        _add_grad_to_knot(s,   dJ_dps, dJ_dvs, dJ_das)
        _add_grad_to_knot(s+1, dJ_dpe, dJ_dve, dJ_dae)
        grad_z[log_Ts_idx] += p.w_smooth * dJ_dTs_jerk * Ts_s

        # ── Sampled costs (aero, bank, speed, acc) ───────────────────────────
        # Evaluate pos/vel/acc at all sample points simultaneously
        H_mat = np.stack([ps, Ts_s*vs, 0.5*Ts_s**2*as_,
                          pe, Ts_s*ve, 0.5*Ts_s**2*ae], axis=1)  # (3,6)
        pos_s  =  H_mat @ h                          # (3, κ+1)
        vel_s  = (H_mat @ dh)  / Ts_s               # (3, κ+1)
        acc_s  = (H_mat @ ddh) / Ts_s**2            # (3, κ+1)

        N = p.kappa + 1
        V2  = np.sum(vel_s**2, axis=0) + 1e-8       # (N,)
        V   = np.sqrt(V2)                            # (N,)
        qbar = q_scale * V2                          # (N,)

        # ── Aerodynamic residual ─────────────────────────────────────────────
        CL_req = p.mass * p.g / (qbar + 1e-8)        # (N,)
        CD_req = p.CD0 + p.k_induced * CL_req**2     # (N,)
        D_force = qbar * CD_req                       # [N] (N,)
        a_drag_mag = D_force / p.mass                 # [m/s²] (N,)

        v_hat = vel_s / (V + 1e-10)                  # (3,N)
        a_along = np.sum(acc_s * v_hat, axis=0)      # (N,)

        drag_res = a_along + a_drag_mag               # (N,)
        lift_res = acc_s[2, :] + p.g                  # (N,)  az + g ≈ 0 trim

        J_aero_vec = drag_res**2 + lift_res**2        # (N,)
        J_total += p.w_aero * np.sum(J_aero_vec)

        # Gradient of aero cost w.r.t. sampled acc_s (partial)
        d_aero_da = np.zeros_like(acc_s)              # (3,N)
        # d/d(acc_s) of drag_res² = 2*drag_res * d(a_along)/d(acc_s)
        # d(a_along)/d(acc_s[:,j]) = v_hat[:,j]
        d_aero_da += 2 * drag_res[np.newaxis,:] * v_hat   # (3,N)
        # d/d(acc_s[2]) of lift_res² = 2*lift_res
        d_aero_da[2, :] += 2 * lift_res               # only z component

        # ── Bank angle  ϕ = atan2(||a_perp||, g) ────────────────────────────
        a_along_full = np.sum(acc_s * v_hat, axis=0)  # (N,)
        a_perp = acc_s - a_along_full[np.newaxis,:] * v_hat  # (3,N)
        a_perp_norm2 = np.sum(a_perp**2, axis=0) + 1e-8  # (N,)
        a_perp_norm  = np.sqrt(a_perp_norm2)              # (N,)
        phi = np.arctan2(a_perp_norm, p.g)                # (N,)

        phi_viol = phi - p.phi_max
        h_bank  = _hinge(phi_viol)
        dh_bank = _hinge_grad(phi_viol)
        J_total += p.w_bank * np.sum(h_bank)

        # d(phi)/d(a_perp_norm) = g / (g²+a_perp_norm²)
        dphi_dapn = p.g / (p.g**2 + a_perp_norm2)    # (N,)
        # d(a_perp_norm)/d(a_perp[:,j]) = a_perp[:,j]/a_perp_norm[j]
        da_perp_da = np.eye(3)[:,:,np.newaxis] - v_hat[:,np.newaxis,:] * v_hat[np.newaxis,:,:]
        # da_perp[:, :, j] is (3,3) for sample j; times dh_bank[j]*dphi_dapn[j]/apn[j]
        bank_chain = (dh_bank * dphi_dapn / a_perp_norm)  # (N,)
        # d(cost)/d(acc_s) += chain * d(||a_perp||)/d(acc_s)
        # d(||a_perp||)/d(acc_s[:,j]) = a_perp[:,j]/a_perp_norm[j]  projected through da_perp/da
        # But da_perp/d(acc_s[:,j]) = (I - v̂v̂ᵀ)[:,j] so:
        d_bank_da = bank_chain[np.newaxis,:] * (a_perp / a_perp_norm[np.newaxis,:])
        # Still need to project through (I - v̂v̂ᵀ):
        vv_corr = np.sum(d_bank_da * v_hat, axis=0)  # (N,)
        d_bank_da -= vv_corr[np.newaxis,:] * v_hat   # projected

        # ── Speed limits ─────────────────────────────────────────────────────
        h_stall = _hinge(p.V_stall - V)
        h_vne   = _hinge(V - p.V_ne)
        J_total += p.w_speed * (np.sum(h_stall) + np.sum(h_vne))

        dh_stall = _hinge_grad(p.V_stall - V) * (-1.0) * (1.0/(2*V))
        dh_vne   = _hinge_grad(V - p.V_ne)  *   1.0  * (1.0/(2*V))
        d_speed_dV2 = p.w_speed * (dh_stall + dh_vne)           # (N,)
        d_speed_dvel = 2 * vel_s * d_speed_dV2[np.newaxis,:]    # (3,N)

        # ── Acceleration limit ───────────────────────────────────────────────
        a2 = np.sum(acc_s**2, axis=0)                # (N,)
        h_acc  = _hinge(a2 - p.a_max**2)
        dh_acc = _hinge_grad(a2 - p.a_max**2)
        J_total += p.w_acc * np.sum(h_acc)
        d_acc_da = 2 * acc_s * (p.w_acc * dh_acc)[np.newaxis,:]  # (3,N)

        # ── Accumulate sampled gradients into decision vars ──────────────────
        # Total gradient w.r.t. acc_s samples
        dJ_da_samp = (p.w_aero * d_aero_da +
                      p.w_bank * d_bank_da +
                      d_acc_da)                        # (3,N)
        # w.r.t. vel_s samples
        dJ_dv_samp = d_speed_dvel                      # (3,N)

        # Propagate through Hermite evaluation:
        # acc_s = (H_mat @ ddh) / Ts²  →  dJ/dH = (dJ/d_acc_s @ ddh.T) / Ts²
        # vel_s = (H_mat @ dh) / Ts    →  dJ/dH += (dJ/d_vel_s @ dh.T) / Ts
        dJ_dH_acc = (dJ_da_samp @ ddh.T) / Ts_s**2   # (3,6)
        dJ_dH_vel = (dJ_dv_samp @ dh.T)  / Ts_s      # (3,6)
        dJ_dH = dJ_dH_acc + dJ_dH_vel                 # (3,6)
        # H = [ps, Ts*vs, 0.5*Ts²*as_, pe, Ts*ve, 0.5*Ts²*ae]
        # Indices: 0=ps,1=Ts*vs,2=0.5Ts²*as_,3=pe,4=Ts*ve,5=0.5Ts²*ae
        dJ_dps2 = dJ_dH[:, 0]
        dJ_dvs2 = dJ_dH[:, 1] * Ts_s
        dJ_das2 = dJ_dH[:, 2] * 0.5 * Ts_s**2
        dJ_dpe2 = dJ_dH[:, 3]
        dJ_dve2 = dJ_dH[:, 4] * Ts_s
        dJ_dae2 = dJ_dH[:, 5] * 0.5 * Ts_s**2

        _add_grad_to_knot(s,   dJ_dps2, dJ_dvs2, dJ_das2)
        _add_grad_to_knot(s+1, dJ_dpe2, dJ_dve2, dJ_dae2)

        # dJ/dTs from sampled costs (state-scaling + coefficient path)
        # state-scaling: d/dTs [vel = dH@dh / Ts] = -vel/Ts
        #                       [acc = dH@ddh / Ts²] = -2*acc/Ts
        dJ_dTs_states = (
            np.sum(dJ_dv_samp * (-vel_s / Ts_s)) +
            np.sum(dJ_da_samp * (-2 * acc_s / Ts_s))
        )
        # coefficient path: d/dTs of H:
        # H[:,1] = Ts*vs  → dH[:,1]/dTs = vs
        # H[:,2] = 0.5Ts²*as → dH[:,2]/dTs = Ts*as
        # H[:,4] = Ts*ve  → dH[:,4]/dTs = ve
        # H[:,5] = 0.5Ts²*ae → dH[:,5]/dTs = Ts*ae
        dJ_dTs_coeff = (
            np.sum(dJ_dH[:, 1] * vs) +
            np.sum(dJ_dH[:, 2] * Ts_s * as_) +
            np.sum(dJ_dH[:, 4] * ve) +
            np.sum(dJ_dH[:, 5] * Ts_s * ae)
        )
        # d(Ts)/d(log_Ts) = Ts
        grad_z[log_Ts_idx] += (dJ_dTs_states + dJ_dTs_coeff) * Ts_s

    return J_total, grad_z


# ═════════════════════════════════════════════════════════════════════════════
#  Build initial decision vector from warm-start
# ═════════════════════════════════════════════════════════════════════════════
def build_warm_start(p: FWParams, start_state, goal_state):
    """
    start_state, goal_state: (9,) arrays [pos(3), vel(3), acc(3)].
    Returns initial z vector.
    """
    M = p.M
    n_int = M - 1
    p_start = start_state[0:3];  v_start = start_state[3:6]
    p_goal  = goal_state[0:3];   v_goal  = goal_state[3:6]

    wpts = dubins_warmstart(p_start, v_start, p_goal, v_goal, M)

    V_nom = 0.5 * (np.linalg.norm(v_start) + np.linalg.norm(v_goal))
    if V_nom < 1e-3:
        V_nom = 15.0

    total_len = sum(np.linalg.norm(wpts[i+1] - wpts[i]) for i in range(M))
    T_nom = max(total_len / V_nom, 1.0)
    T_seg = T_nom / M

    # Nominal mid-segment velocity direction
    v_mid = 0.5*(v_start + v_goal)
    v_mid_n = v_mid / (np.linalg.norm(v_mid) + 1e-8) * V_nom

    z0 = np.zeros(n_int*9 + M)
    for i in range(1, M):   # interior knots 1..M-1
        idx = (i-1)*9
        T_bar = T_seg        # initial T̄ ≈ T_seg
        z0[idx:idx+3]   = wpts[i]             # position
        z0[idx+3:idx+6] = T_bar * v_mid_n     # scaled velocity v̂ = T̄·v
        z0[idx+6:idx+9] = 0.0                 # scaled acceleration (zero)

    # log segment durations
    z0[n_int*9:] = np.log(T_seg) * np.ones(M)
    return z0


# ═════════════════════════════════════════════════════════════════════════════
#  Trajectory evaluation (dense sampling for plotting)
# ═════════════════════════════════════════════════════════════════════════════
def evaluate_trajectory(knots, Ts_arr, n_per_seg=100):
    """
    Returns arrays: t_arr, pos, vel, acc, speed, bank_angle
    """
    M = len(Ts_arr)
    t_list, pos_list, vel_list, acc_list = [], [], [], []
    t_offset = 0.0
    for s in range(M):
        tau_arr = np.linspace(0, 1, n_per_seg, endpoint=(s == M-1))
        ks = knots[s];   ke = knots[s+1]
        pos, vel, acc = _eval_segment(ks[0:3], ks[3:6], ks[6:9],
                                       ke[0:3], ke[3:6], ke[6:9],
                                       Ts_arr[s], tau_arr)
        t_arr = t_offset + tau_arr * Ts_arr[s]
        t_list.append(t_arr)
        pos_list.append(pos)
        vel_list.append(vel)
        acc_list.append(acc)
        t_offset += Ts_arr[s]

    t   = np.concatenate(t_list)
    pos = np.concatenate(pos_list, axis=1)
    vel = np.concatenate(vel_list, axis=1)
    acc = np.concatenate(acc_list, axis=1)

    V  = np.linalg.norm(vel, axis=0)
    V2 = V**2 + 1e-8
    v_hat = vel / (V + 1e-10)
    a_along = np.sum(acc * v_hat, axis=0)
    a_perp  = acc - a_along[np.newaxis,:] * v_hat
    a_perp_n = np.linalg.norm(a_perp, axis=0)
    phi = np.degrees(np.arctan2(a_perp_n, 9.81))

    return t, pos, vel, acc, V, phi


# ═════════════════════════════════════════════════════════════════════════════
#  Planner class
# ═════════════════════════════════════════════════════════════════════════════
class FWHermitePlanner:
    def __init__(self, params: FWParams = None):
        self.p = params or FWParams()

    def plan(self, start_state: np.ndarray, goal_state: np.ndarray):
        """
        start_state, goal_state: (9,) = [pos(3), vel(3), acc(3)]
        Returns dict with knots, Ts, dense trajectory arrays.
        """
        p = self.p
        z0 = build_warm_start(p, start_state, goal_state)

        print(f"[MIGHTY-FW]  M={p.M} segments, "
              f"{len(z0)} decision variables")
        print(f"[MIGHTY-FW]  Warm-start  J0 = "
              f"{cost_and_grad(z0, p, start_state, goal_state)[0]:.4g}")
        print(f"[MIGHTY-FW]  Optimising …")
        t0 = time.time()

        def fun_and_grad(z):
            return cost_and_grad(z, p, start_state, goal_state)

        res = minimize(
            fun_and_grad,
            z0,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": p.max_iter,
                     "ftol": p.tol,
                     "gtol": p.gtol,
                     "maxls": 40},
        )
        elapsed = time.time() - t0
        print(f"[MIGHTY-FW]  Done in {elapsed:.2f}s  |  "
              f"J* = {res.fun:.4g}  |  "
              f"{'CONVERGED' if res.success else 'NOT CONVERGED: ' + res.message}")

        z_opt = res.x
        M = p.M
        n_int = M - 1
        knots, Ts, _ = _unpack(z_opt, M, start_state, goal_state)

        t_arr, pos, vel, acc, speed, bank = evaluate_trajectory(knots, Ts)

        total_time = np.sum(Ts)
        path_len   = np.trapezoid(speed, t_arr)
        print(f"[MIGHTY-FW]  Total time  = {total_time:.2f} s")
        print(f"[MIGHTY-FW]  Path length = {path_len:.1f} m")
        print(f"[MIGHTY-FW]  Max speed   = {np.max(speed):.1f} m/s")
        print(f"[MIGHTY-FW]  Max bank    = {np.max(bank):.1f}°  "
              f"(limit {np.degrees(p.phi_max):.0f}°)")
        print(f"[MIGHTY-FW]  Segment T   = {np.round(Ts, 2)}")

        return {
            "success": res.success,
            "z_opt":   z_opt,
            "knots":   knots,
            "Ts":      Ts,
            "t":       t_arr,
            "pos":     pos,
            "vel":     vel,
            "acc":     acc,
            "speed":   speed,
            "bank":    bank,
            "J_opt":   res.fun,
            "elapsed": elapsed,
        }


# ═════════════════════════════════════════════════════════════════════════════
#  Visualiser
# ═════════════════════════════════════════════════════════════════════════════
def plot_trajectory(result: dict, params: FWParams,
                    start_state, goal_state, title="FW Hermite Trajectory"):
    knots   = result["knots"]
    Ts      = result["Ts"]
    t       = result["t"]
    pos     = result["pos"]
    speed   = result["speed"]
    bank    = result["bank"]
    acc_arr = result["acc"]
    M = params.M

    knot_pos = np.array([k[0:3] for k in knots])   # (M+1, 3)
    total_time = np.sum(Ts)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── 3-D trajectory ───────────────────────────────────────────────────────
    ax3d = fig.add_subplot(gs[0:2, 0:2], projection='3d')
    sc = ax3d.scatter(pos[0], pos[1], pos[2],
                      c=speed, cmap='plasma', s=1.5, zorder=2)
    ax3d.plot(*knot_pos.T, 'o', color='cyan', ms=7,
              markeredgecolor='k', lw=0, label='Knots', zorder=5)
    ax3d.scatter(*start_state[0:3], marker='^', s=120,
                 color='lime',   zorder=6, label='Start')
    ax3d.scatter(*goal_state[0:3],  marker='*', s=180,
                 color='red',    zorder=6, label='Goal')

    # Velocity arrows at knots
    for k in knots:
        v = k[3:6]
        ax3d.quiver(k[0], k[1], k[2],
                    v[0], v[1], v[2],
                    length=3.0, normalize=True,
                    color='deepskyblue', alpha=0.7)

    cbar = fig.colorbar(sc, ax=ax3d, shrink=0.55, pad=0.1)
    cbar.set_label('Speed [m/s]')
    ax3d.set_xlabel('X [m]'); ax3d.set_ylabel('Y [m]'); ax3d.set_zlabel('Z [m]')
    ax3d.set_title('3-D Trajectory  (colour = speed)')
    ax3d.legend(fontsize=8)

    # ── Top-down (X-Y) view ──────────────────────────────────────────────────
    ax_xy = fig.add_subplot(gs[2, 0])
    ax_xy.scatter(pos[0], pos[1], c=speed, cmap='plasma', s=1)
    ax_xy.plot(knot_pos[:,0], knot_pos[:,1], 'o-',
               color='cyan', ms=5, lw=1, label='Knots')
    ax_xy.set_xlabel('X [m]'); ax_xy.set_ylabel('Y [m]')
    ax_xy.set_title('Top-down (X–Y)')
    ax_xy.set_aspect('equal'); ax_xy.grid(True, alpha=0.3)

    # ── Side (X-Z) view ──────────────────────────────────────────────────────
    ax_xz = fig.add_subplot(gs[2, 1])
    ax_xz.scatter(pos[0], pos[2], c=speed, cmap='plasma', s=1)
    ax_xz.plot(knot_pos[:,0], knot_pos[:,2], 'o-',
               color='cyan', ms=5, lw=1)
    ax_xz.set_xlabel('X [m]'); ax_xz.set_ylabel('Z [m]')
    ax_xz.set_title('Side (X–Z)')
    ax_xz.grid(True, alpha=0.3)

    # ── Speed ────────────────────────────────────────────────────────────────
    ax_v = fig.add_subplot(gs[0, 2])
    ax_v.plot(t, speed, 'b', lw=1.5)
    ax_v.axhline(params.V_stall, color='r', ls='--', lw=1, label=f'V_stall={params.V_stall}')
    ax_v.axhline(params.V_ne,    color='m', ls='--', lw=1, label=f'V_ne={params.V_ne}')
    for s_start in np.cumsum(np.concatenate([[0], Ts[:-1]])):
        ax_v.axvline(s_start, color='gray', ls=':', lw=0.8, alpha=0.6)
    ax_v.set_xlabel('Time [s]'); ax_v.set_ylabel('Speed [m/s]')
    ax_v.set_title('Airspeed')
    ax_v.legend(fontsize=7); ax_v.grid(True, alpha=0.3)

    # ── Bank angle ───────────────────────────────────────────────────────────
    ax_phi = fig.add_subplot(gs[1, 2])
    ax_phi.plot(t, bank, 'darkorange', lw=1.5)
    ax_phi.axhline(np.degrees(params.phi_max), color='r', ls='--', lw=1,
                   label=f'φ_max={np.degrees(params.phi_max):.0f}°')
    for s_start in np.cumsum(np.concatenate([[0], Ts[:-1]])):
        ax_phi.axvline(s_start, color='gray', ls=':', lw=0.8, alpha=0.6)
    ax_phi.set_xlabel('Time [s]'); ax_phi.set_ylabel('Bank angle [°]')
    ax_phi.set_title('Bank angle (coordinated turn)')
    ax_phi.legend(fontsize=7); ax_phi.grid(True, alpha=0.3)

    # ── Acceleration magnitude ───────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[2, 2])
    a_mag = np.linalg.norm(acc_arr, axis=0)
    ax_a.plot(t, a_mag, 'green', lw=1.5)
    ax_a.axhline(params.a_max, color='r', ls='--', lw=1,
                 label=f'a_max={params.a_max} m/s²')
    ax_a.set_xlabel('Time [s]'); ax_a.set_ylabel('|a| [m/s²]')
    ax_a.set_title('Acceleration magnitude')
    ax_a.legend(fontsize=7); ax_a.grid(True, alpha=0.3)

    # ── Segment duration breakdown ───────────────────────────────────────────
    ax_T = fig.add_subplot(gs[0, 0])
    segs = np.arange(M)
    ax_T.bar(segs, Ts, color='steelblue', edgecolor='k')
    ax_T.set_xlabel('Segment'); ax_T.set_ylabel('Duration [s]')
    ax_T.set_title(f'Segment durations  (Σ={total_time:.1f}s)')
    ax_T.set_xticks(segs)
    ax_T.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  Demo / entry point
# ═════════════════════════════════════════════════════════════════════════════
def build_demo_scenario():
    """
    Scenario: fixed-wing UAV climbs from origin heading east,
    turns and descends to a goal 300 m away heading south-east.
    """
    # Start: position (0,0,100)m, V=15 m/s heading east, level flight
    V0 = 15.0
    p_start = np.array([0.0, 0.0, 100.0])
    v_start = np.array([V0,  0.0,   0.0])
    a_start = np.array([0.0, 0.0,   0.0])
    start_state = np.concatenate([p_start, v_start, a_start])

    # Goal: position (250, -150, 80)m, V=18 m/s heading south-east
    V1 = 18.0
    heading = math.radians(-45)    # south-east
    p_goal  = np.array([250.0, -150.0, 80.0])
    v_goal  = np.array([V1*math.cos(heading), V1*math.sin(heading), 0.0])
    a_goal  = np.array([0.0, 0.0, 0.0])
    goal_state = np.concatenate([p_goal, v_goal, a_goal])

    return start_state, goal_state


def main():
    parser = argparse.ArgumentParser(
        description="Fixed-wing Hermite spline trajectory optimiser")
    parser.add_argument("--M",        type=int,   default=7,    help="Number of spline segments")
    parser.add_argument("--kappa",    type=int,   default=10,   help="Samples per segment")
    parser.add_argument("--w_time",   type=float, default=5e1)
    parser.add_argument("--w_smooth", type=float, default=1e-1)
    parser.add_argument("--w_aero",   type=float, default=5e1)
    parser.add_argument("--w_bank",   type=float, default=1e3)
    parser.add_argument("--w_speed",  type=float, default=1e3)
    parser.add_argument("--w_acc",    type=float, default=5e2)
    parser.add_argument("--phi_max",  type=float, default=55.0, help="Max bank angle [deg]")
    parser.add_argument("--V_stall",  type=float, default=8.0)
    parser.add_argument("--V_ne",     type=float, default=30.0)
    parser.add_argument("--V_start",  type=float, default=15.0)
    parser.add_argument("--V_goal",   type=float, default=18.0)
    parser.add_argument("--max_iter", type=int,   default=800)
    parser.add_argument("--save",     type=str,   default=None, help="Save figure to file")
    args = parser.parse_args()

    p = FWParams(
        M        = args.M,
        kappa    = args.kappa,
        w_time   = args.w_time,
        w_smooth = args.w_smooth,
        w_aero   = args.w_aero,
        w_bank   = args.w_bank,
        w_speed  = args.w_speed,
        w_acc    = args.w_acc,
        phi_max  = math.radians(args.phi_max),
        V_stall  = args.V_stall,
        V_ne     = args.V_ne,
        max_iter = args.max_iter,
    )

    start_state, goal_state = build_demo_scenario()
    # Override speeds if specified
    v_dir_s = start_state[3:6] / np.linalg.norm(start_state[3:6])
    v_dir_g = goal_state[3:6]  / np.linalg.norm(goal_state[3:6])
    start_state[3:6] = args.V_start * v_dir_s
    goal_state[3:6]  = args.V_goal  * v_dir_g

    print("=" * 60)
    print("  Fixed-Wing Hermite Trajectory Optimiser (MIGHTY-style)")
    print("=" * 60)
    print(f"  Start : p={start_state[0:3]}  v={start_state[3:6]}")
    print(f"  Goal  : p={goal_state[0:3]}  v={goal_state[3:6]}")
    print(f"  M={p.M}, κ={p.kappa}")
    print("=" * 60)

    planner = FWHermitePlanner(p)
    result  = planner.plan(start_state, goal_state)

    fig = plot_trajectory(result, p, start_state, goal_state)
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"[MIGHTY-FW]  Figure saved → {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
