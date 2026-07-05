/**
 * fw_hermite_planner.cpp
 *
 * Fixed-Wing Hermite Spline Trajectory Optimizer  —  MIGHTY-style
 * ----------------------------------------------------------------
 * See fw_hermite_planner.hpp for full documentation.
 *
 * Cost function derivation
 * ========================
 *
 *  Fixed-wing point-mass model (3-DOF)
 *  ------------------------------------
 *  We treat the aircraft as a point mass with speed V = ||v||.
 *  The aerodynamic forces (in wind frame) are:
 *
 *    L = ½ ρ V² S_wing CL(α)        (lift, perpendicular to v)
 *    D = ½ ρ V² S_wing CD(α)        (drag, parallel to -v)
 *    CL(α) = CL0 + CLa·α
 *    CD(α) = CD0 + k·CL²
 *
 *  Equations of motion (body fixed):
 *    m·a = L·n̂_lift + D·(-v̂) + m·g·ẑ
 *
 *  Where n̂_lift ⊥ v̂ and lies in the lift plane (the plane spanned by v̂
 *  and the body z-axis).  For a banked turn in 3-D:
 *
 *    Bank angle φ:  cos(φ) = g / ||a_centripetal||  (coordinated turn)
 *    φ = atan2(||a_centripetal||, g)
 *    a_centripetal = a - (a·v̂)v̂  (component of accel perp. to velocity)
 *
 *  Aerodynamic residual cost (MIGHTY-style sampled cost):
 *    ℓ_aero = ||m·a - F_aero(v)||²
 *  where F_aero is the expected total aerodynamic force given the current
 *  speed. We use the trimmed condition F_trim = m*g (level flight) as the
 *  nominal, penalising deviations in the lift-drag balance.
 *
 *  Decision variables
 *  ------------------
 *  For M segments, M+1 knots; boundary knots 0 and M are fixed.
 *  Interior knots i=1..M-1:  scaled derivatives v̂ᵢ, âᵢ, position pᵢ
 *  Segment durations: σs  (Ts = exp(σs))
 *
 *  Total: (M-1)*9  +  M  scalar DOF
 *
 *  Hermite→Bézier map (MIGHTY eq. 2)
 *  -----------------------------------
 *  c₀ = p_s
 *  c₁ = p_s + (Ts/5) v_s
 *  c₂ = p_s + (2Ts/5) v_s + (Ts²/20) a_s
 *  c₃ = p_e - (2Ts/5) v_e + (Ts²/20) a_e
 *  c₄ = p_e - (Ts/5) v_e
 *  c₅ = p_e
 *
 *  Jerk integral (closed form, MIGHTY eq. 6)
 *  ------------------------------------------
 *  J_smooth,s = Cs · Δs^T Q Δs
 *  Δs,m = c_{m+3} - 3c_{m+2} + 3c_{m+1} - c_m,  m=0,1,2
 *  Cs   = 3600 · Ts^{-5}
 *  G matrix (Bernstein-quadratic Gram):
 *    G00=G22=1/5, G11=2/15, G01=G12=1/10, G02=1/30
 */

#include "fw_hermite_planner.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>

namespace fw_hermite {

// ─────────────────────────────────────────────────────────────────────────────
//  Constructor
// ─────────────────────────────────────────────────────────────────────────────
FWHermitePlanner::FWHermitePlanner(const FWParams& params)
    : p_(params) {}

// ─────────────────────────────────────────────────────────────────────────────
//  Smooth hinge  ϕ_μ(x) = (1/μ) log(1 + exp(μ·x))
// ─────────────────────────────────────────────────────────────────────────────
casadi::MX FWHermitePlanner::smooth_hinge(const casadi::MX& x) const {
    const double mu = 50.0;
    return (1.0 / mu) * casadi::MX::log(1.0 + casadi::MX::exp(mu * x));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Hermite → Bézier  (MIGHTY eq. 2, symbolic)
// ─────────────────────────────────────────────────────────────────────────────
std::vector<casadi::MX> FWHermitePlanner::hermite_to_bezier(
    const casadi::MX& ps, const casadi::MX& vs, const casadi::MX& as,
    const casadi::MX& pe, const casadi::MX& ve, const casadi::MX& ae,
    const casadi::MX& Ts) const
{
    std::vector<casadi::MX> c(6);
    c[0] = ps;
    c[1] = ps + (Ts / 5.0)  * vs;
    c[2] = ps + (2.0*Ts/5.0)* vs + (Ts*Ts/20.0) * as;
    c[3] = pe - (2.0*Ts/5.0)* ve + (Ts*Ts/20.0) * ae;
    c[4] = pe - (Ts / 5.0)  * ve;
    c[5] = pe;
    return c;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Closed-form integrated squared jerk on one Bézier segment
//  (MIGHTY eq. 6 — scalar surrogate with diagonal G weights)
// ─────────────────────────────────────────────────────────────────────────────
casadi::MX FWHermitePlanner::bezier_jerk_integral(
    const std::vector<casadi::MX>& c, const casadi::MX& Ts) const
{
    // Forward differences Δm = c[m+3] - 3c[m+2] + 3c[m+1] - c[m]
    std::vector<casadi::MX> delta(3);
    for (int m = 0; m < 3; ++m)
        delta[m] = c[m+3] - 3.0*c[m+2] + 3.0*c[m+1] - c[m];

    // G diagonal weights  {G00, G11, G22} = {1/5, 2/15, 1/5}
    const double w[3] = {0.2, 2.0/15.0, 0.2};

    casadi::MX J = casadi::MX::zeros(1);
    for (int m = 0; m < 3; ++m)
        J += w[m] * casadi::MX::dot(delta[m], delta[m]);

    // Cs = 3600 * Ts^{-5}
    casadi::MX Cs = 3600.0 * casadi::MX::pow(Ts, -5.0);
    return Cs * J;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Bank angle from velocity and acceleration vectors (symbolic)
//  Coordinated-turn model: φ = atan2(||a_perp||, g)
//  where a_perp = a - (a·v̂) v̂  is the centripetal component
// ─────────────────────────────────────────────────────────────────────────────
casadi::MX FWHermitePlanner::bank_angle_sym(const casadi::MX& vel,
                                            const casadi::MX& acc) const
{
    casadi::MX V2  = casadi::MX::dot(vel, vel) + 1e-6;
    casadi::MX a_along = casadi::MX::dot(acc, vel) / V2;
    casadi::MX a_perp  = acc - a_along * vel;
    casadi::MX a_perp2 = casadi::MX::dot(a_perp, a_perp);
    return casadi::MX::atan2(casadi::MX::sqrt(a_perp2 + 1e-8),
                             casadi::MX(p_.g));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Jerk smoothness cost  (closed-form over all segments)
// ─────────────────────────────────────────────────────────────────────────────
casadi::MX FWHermitePlanner::build_jerk_cost(
    const std::vector<casadi::MX>& knots,   // (M+1) × [p;v;a] each 9×1
    const casadi::MX& T_sigma) const
{
    casadi::MX J_smooth = casadi::MX::zeros(1);
    for (int s = 0; s < p_.M; ++s) {
        casadi::MX Ts = casadi::MX::exp(T_sigma(s));
        auto ps = knots[s  ](casadi::Slice(0,3));
        auto vs = knots[s  ](casadi::Slice(3,6));
        auto as = knots[s  ](casadi::Slice(6,9));
        auto pe = knots[s+1](casadi::Slice(0,3));
        auto ve = knots[s+1](casadi::Slice(3,6));
        auto ae = knots[s+1](casadi::Slice(6,9));
        auto bez = hermite_to_bezier(ps, vs, as, pe, ve, ae, Ts);
        J_smooth += bezier_jerk_integral(bez, Ts);
    }
    return J_smooth;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Aerodynamic residual cost  (sampled)
//  Penalises: ||m·a_sample - F_aero_model(v_sample)||²
//
//  F_aero = [-D · v̂ + L · n̂_lift] + m·g·ẑ   (net force in world frame)
//  For the cost we use a simplified trim-based residual:
//    ℓ_aero = ||a_z + g||² + ||D_excess||²
//  where D_excess = m·(a·v̂) + D_trim  and D_trim = ½ρV² S CD(CL_trim)
// ─────────────────────────────────────────────────────────────────────────────
casadi::MX FWHermitePlanner::build_aero_cost(
    const std::vector<casadi::MX>& knots,
    const casadi::MX& T_sigma) const
{
    casadi::MX J_aero = casadi::MX::zeros(1);
    const double q_scale = 0.5 * p_.rho * p_.S_wing;

    for (int s = 0; s < p_.M; ++s) {
        casadi::MX Ts = casadi::MX::exp(T_sigma(s));
        auto ps = knots[s  ](casadi::Slice(0,3));
        auto vs = knots[s  ](casadi::Slice(3,6));
        auto as = knots[s  ](casadi::Slice(6,9));
        auto pe = knots[s+1](casadi::Slice(0,3));
        auto ve = knots[s+1](casadi::Slice(3,6));
        auto ae = knots[s+1](casadi::Slice(6,9));

        // Sample kappa points
        for (int j = 0; j <= p_.kappa; ++j) {
            double tau = static_cast<double>(j) / p_.kappa;

            // Quintic Hermite basis values (numeric coefficients)
            double t = tau, t2=t*t, t3=t2*t, t4=t3*t, t5=t4*t;
            double h0 = 1-10*t3+15*t4-6*t5;
            double h1 = t-6*t3+8*t4-3*t5;
            double h2 = 0.5*t2-1.5*t3+1.5*t4-0.5*t5;
            double h3 = 10*t3-15*t4+6*t5;
            double h4 = -4*t3+7*t4-3*t5;
            double h5 = 0.5*t3-t4+0.5*t5;

            // Basis derivatives  (scaled by 1/Ts for physical velocity)
            double dh0 = -30*t2+60*t3-30*t4;
            double dh1 = 1-18*t2+32*t3-15*t4;
            double dh2 = t-4.5*t2+6*t3-2.5*t4;
            double dh3 = 30*t2-60*t3+30*t4;
            double dh4 = -12*t2+28*t3-15*t4;
            double dh5 = 1.5*t2-4*t3+2.5*t4;

            // Second derivatives (scaled by 1/Ts^2)
            double ddh0 = -60*t+180*t2-120*t3;
            double ddh1 = -36*t+96*t2-60*t3;
            double ddh2 = 1-9*t+18*t2-10*t3;
            double ddh3 = 60*t-180*t2+120*t3;
            double ddh4 = -24*t+84*t2-60*t3;
            double ddh5 = 3*t-12*t2+10*t3;

            // Symbolic expressions for position, velocity, accel at sample
            casadi::MX p_samp =
                h0*ps + h1*(Ts*vs) + h2*(Ts*Ts*0.5*as) +
                h3*pe + h4*(Ts*ve) + h5*(Ts*Ts*0.5*ae);
            casadi::MX v_samp =
                (dh0*ps + dh1*(Ts*vs) + dh2*(Ts*Ts*0.5*as) +
                 dh3*pe + dh4*(Ts*ve) + dh5*(Ts*Ts*0.5*ae)) / Ts;
            casadi::MX a_samp =
                (ddh0*ps + ddh1*(Ts*vs) + ddh2*(Ts*Ts*0.5*as) +
                 ddh3*pe + ddh4*(Ts*ve) + ddh5*(Ts*Ts*0.5*ae)) / (Ts*Ts);

            // Speed and dynamic pressure
            casadi::MX V2   = casadi::MX::dot(v_samp, v_samp) + 1e-6;
            casadi::MX V    = casadi::MX::sqrt(V2);
            casadi::MX qbar = q_scale * V2;   // ½ρV²·S

            // CL needed for level flight: L = mg  → CL_req = mg/(qbar)
            casadi::MX CL_req = p_.mass * p_.g / (qbar + 1e-6);
            casadi::MX CD_req = p_.CD0 + p_.k_induced * CL_req * CL_req;

            // Expected drag deceleration (along −v̂)
            casadi::MX D_force = qbar * CD_req;  // [N]
            casadi::MX a_drag_mag = D_force / p_.mass;

            // Along-path acceleration from trajectory
            casadi::MX v_hat = v_samp / (V + 1e-8);
            casadi::MX a_along = casadi::MX::dot(a_samp, v_hat);

            // Residual: drag excess  (should be ~ -a_drag_mag in trim)
            casadi::MX drag_residual = a_along + a_drag_mag;

            // Vertical: a_z + g should ≈ 0 in trim
            // (lift balances gravity)
            casadi::MX az = a_samp(2);
            casadi::MX lift_residual = az + p_.g;

            J_aero += drag_residual * drag_residual
                    + lift_residual * lift_residual;
        }
    }
    return J_aero;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Bank angle cost  (sampled, soft hinge)
// ─────────────────────────────────────────────────────────────────────────────
casadi::MX FWHermitePlanner::build_bank_cost(
    const std::vector<casadi::MX>& knots,
    const casadi::MX& T_sigma) const
{
    casadi::MX J_bank = casadi::MX::zeros(1);
    for (int s = 0; s < p_.M; ++s) {
        casadi::MX Ts = casadi::MX::exp(T_sigma(s));
        auto ps = knots[s  ](casadi::Slice(0,3));
        auto vs = knots[s  ](casadi::Slice(3,6));
        auto as = knots[s  ](casadi::Slice(6,9));
        auto pe = knots[s+1](casadi::Slice(0,3));
        auto ve = knots[s+1](casadi::Slice(3,6));
        auto ae = knots[s+1](casadi::Slice(6,9));

        for (int j = 0; j <= p_.kappa; ++j) {
            double tau = static_cast<double>(j) / p_.kappa;
            double t = tau, t2=t*t, t3=t2*t, t4=t3*t, t5=t4*t;
            double dh0 = -30*t2+60*t3-30*t4;
            double dh1 = 1-18*t2+32*t3-15*t4;
            double dh2 = t-4.5*t2+6*t3-2.5*t4;
            double dh3 = 30*t2-60*t3+30*t4;
            double dh4 = -12*t2+28*t3-15*t4;
            double dh5 = 1.5*t2-4*t3+2.5*t4;
            double ddh0 = -60*t+180*t2-120*t3;
            double ddh1 = -36*t+96*t2-60*t3;
            double ddh2 = 1-9*t+18*t2-10*t3;
            double ddh3 = 60*t-180*t2+120*t3;
            double ddh4 = -24*t+84*t2-60*t3;
            double ddh5 = 3*t-12*t2+10*t3;

            casadi::MX v_samp =
                (dh0*ps + dh1*(Ts*vs) + dh2*(Ts*Ts*0.5*as) +
                 dh3*pe + dh4*(Ts*ve) + dh5*(Ts*Ts*0.5*ae)) / Ts;
            casadi::MX a_samp =
                (ddh0*ps + ddh1*(Ts*vs) + ddh2*(Ts*Ts*0.5*as) +
                 ddh3*pe + ddh4*(Ts*ve) + ddh5*(Ts*Ts*0.5*ae)) / (Ts*Ts);

            casadi::MX phi = bank_angle_sym(v_samp, a_samp);
            J_bank += smooth_hinge(phi - p_.phi_max);
        }
    }
    return J_bank;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Speed limit cost  (sampled, soft hinge — stall and VNE)
// ─────────────────────────────────────────────────────────────────────────────
casadi::MX FWHermitePlanner::build_speed_cost(
    const std::vector<casadi::MX>& knots,
    const casadi::MX& T_sigma) const
{
    casadi::MX J_speed = casadi::MX::zeros(1);
    for (int s = 0; s < p_.M; ++s) {
        casadi::MX Ts = casadi::MX::exp(T_sigma(s));
        auto ps = knots[s  ](casadi::Slice(0,3));
        auto vs = knots[s  ](casadi::Slice(3,6));
        auto as = knots[s  ](casadi::Slice(6,9));
        auto pe = knots[s+1](casadi::Slice(0,3));
        auto ve = knots[s+1](casadi::Slice(3,6));
        auto ae = knots[s+1](casadi::Slice(6,9));

        for (int j = 0; j <= p_.kappa; ++j) {
            double tau = static_cast<double>(j) / p_.kappa;
            double t = tau, t2=t*t, t3=t2*t, t4=t3*t, t5=t4*t;
            double dh0 = -30*t2+60*t3-30*t4;
            double dh1 = 1-18*t2+32*t3-15*t4;
            double dh2 = t-4.5*t2+6*t3-2.5*t4;
            double dh3 = 30*t2-60*t3+30*t4;
            double dh4 = -12*t2+28*t3-15*t4;
            double dh5 = 1.5*t2-4*t3+2.5*t4;

            casadi::MX v_samp =
                (dh0*ps + dh1*(Ts*vs) + dh2*(Ts*Ts*0.5*as) +
                 dh3*pe + dh4*(Ts*ve) + dh5*(Ts*Ts*0.5*ae)) / Ts;
            casadi::MX V2 = casadi::MX::dot(v_samp, v_samp);
            casadi::MX V  = casadi::MX::sqrt(V2 + 1e-8);

            J_speed += smooth_hinge(p_.V_stall - V);    // below stall
            J_speed += smooth_hinge(V - p_.V_ne);       // above VNE
        }
    }
    return J_speed;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Acceleration magnitude cost  (sampled, soft hinge)
// ─────────────────────────────────────────────────────────────────────────────
casadi::MX FWHermitePlanner::build_acc_cost(
    const std::vector<casadi::MX>& knots,
    const casadi::MX& T_sigma) const
{
    casadi::MX J_acc = casadi::MX::zeros(1);
    for (int s = 0; s < p_.M; ++s) {
        casadi::MX Ts = casadi::MX::exp(T_sigma(s));
        auto ps = knots[s  ](casadi::Slice(0,3));
        auto vs = knots[s  ](casadi::Slice(3,6));
        auto as = knots[s  ](casadi::Slice(6,9));
        auto pe = knots[s+1](casadi::Slice(0,3));
        auto ve = knots[s+1](casadi::Slice(3,6));
        auto ae = knots[s+1](casadi::Slice(6,9));

        for (int j = 0; j <= p_.kappa; ++j) {
            double tau = static_cast<double>(j) / p_.kappa;
            double t = tau, t2=t*t, t3=t2*t, t4=t3*t, t5=t4*t;
            double ddh0 = -60*t+180*t2-120*t3;
            double ddh1 = -36*t+96*t2-60*t3;
            double ddh2 = 1-9*t+18*t2-10*t3;
            double ddh3 = 60*t-180*t2+120*t3;
            double ddh4 = -24*t+84*t2-60*t3;
            double ddh5 = 3*t-12*t2+10*t3;

            casadi::MX a_samp =
                (ddh0*ps + ddh1*(Ts*vs) + ddh2*(Ts*Ts*0.5*as) +
                 ddh3*pe + ddh4*(Ts*ve) + ddh5*(Ts*Ts*0.5*ae)) / (Ts*Ts);
            casadi::MX a2 = casadi::MX::dot(a_samp, a_samp);
            J_acc += smooth_hinge(a2 - p_.a_max * p_.a_max);
        }
    }
    return J_acc;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Dubins warm-start
//  2-D RSR Dubins path, then linearly interpolate altitude.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<Eigen::Vector3d> dubins_waypoints(
    const KnotState& start, const KnotState& goal,
    int M, double R)
{
    // Extract 2-D headings from velocity vectors
    auto yaw = [](const Eigen::Vector3d& v) {
        return std::atan2(v(1), v(0));
    };
    double xi = start.p(0), yi = start.p(1);
    double xf = goal.p(0),  yf = goal.p(1);
    double hi  = yaw(start.v);
    double hf  = yaw(goal.v);

    // Centres of turning circles (right turn: +π/2 offset)
    auto cx_r = [](double x, double h, double r){ return x + r*std::sin(h); };
    auto cy_r = [](double y, double h, double r){ return y - r*std::cos(h); };
    double cx1 = cx_r(xi, hi, R),  cy1 = cy_r(yi, hi, R);
    double cx2 = cx_r(xf, hf, R),  cy2 = cy_r(yf, hf, R);

    // RSR: straight segment between tangent points
    double dx = cx2 - cx1, dy = cy2 - cy1;
    double dist = std::sqrt(dx*dx + dy*dy);
    double beta = std::atan2(dy, dx);

    // Tangent points
    double tx1 = cx1 + R*std::cos(beta - M_PI/2.0);
    double ty1 = cy1 + R*std::sin(beta - M_PI/2.0);
    double tx2 = cx2 + R*std::cos(beta - M_PI/2.0);
    double ty2 = cy2 + R*std::sin(beta - M_PI/2.0);

    // Build total arc-length parametric path
    // Arc1: start → tx1,  Straight: tx1→tx2,  Arc2: tx2→goal
    double z0 = start.p(2), z1 = goal.p(2);

    // Uniformly sample M+1 points along the Dubins path
    double total = dist + R*1.5; // approximate
    std::vector<Eigen::Vector3d> pts(M+1);
    for (int i = 0; i <= M; ++i) {
        double frac = static_cast<double>(i) / M;
        // Lerp in 2-D along the chord (simple fallback)
        // For a real system replace with proper arc integration
        double x = xi + frac*(xf - xi);
        double y = yi + frac*(yf - yi);
        // Add lateral offset via tangent waypoints
        double blend = 4.0 * frac * (1.0 - frac); // bump in middle
        x += blend * 0.3 * (tx1 - 0.5*(xi+xf));
        y += blend * 0.3 * (ty1 - 0.5*(yi+yf));
        double z = z0 + frac*(z1 - z0);
        pts[i] = Eigen::Vector3d(x, y, z);
    }
    // Fix endpoints exactly
    pts[0] = start.p;
    pts[M] = goal.p;
    return pts;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Warm-start decision vector
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> FWHermitePlanner::warm_start(
    const KnotState& start, const KnotState& goal) const
{
    auto wpts = dubins_waypoints(start, goal, p_.M,
                                 /*turn_radius=*/20.0);
    double V_nom = 0.5*(start.v.norm() + goal.v.norm());
    if (V_nom < 1e-3) V_nom = 15.0;

    // Total path length → nominal time per segment
    double total_len = 0.0;
    for (int i = 0; i < p_.M; ++i)
        total_len += (wpts[i+1] - wpts[i]).norm();
    double T_nom = std::max(total_len / V_nom, 1.0);
    double T_seg = T_nom / p_.M;

    int n_interior = p_.M - 1;
    int n_z = n_interior * 9 + p_.M;
    std::vector<double> z0(n_z, 0.0);

    // Fill interior knot states
    Eigen::Vector3d v_mid = 0.5*(start.v + goal.v).normalized() * V_nom;
    for (int i = 0; i < n_interior; ++i) {
        int idx = i * 9;
        // Position
        z0[idx+0] = wpts[i+1](0);
        z0[idx+1] = wpts[i+1](1);
        z0[idx+2] = wpts[i+1](2);
        // Scaled velocity v̂ = T̄·v
        double T_bar = T_seg;
        z0[idx+3] = T_bar * v_mid(0);
        z0[idx+4] = T_bar * v_mid(1);
        z0[idx+5] = T_bar * v_mid(2);
        // Scaled acceleration (zero init)
        z0[idx+6] = 0.0;
        z0[idx+7] = 0.0;
        z0[idx+8] = 0.0;
    }
    // Segment durations  σ = log(Ts)
    for (int s = 0; s < p_.M; ++s)
        z0[n_interior*9 + s] = std::log(T_seg);

    return z0;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Main plan()  —  builds and solves the CasADi NLP
// ─────────────────────────────────────────────────────────────────────────────
FWTrajectory FWHermitePlanner::plan(const KnotState& start,
                                    const KnotState& goal)
{
    using namespace casadi;
    int M = p_.M;
    int n_interior = M - 1;

    // ── Decision variables ───────────────────────────────────────────────────
    // z = [p1,v1_hat,a1_hat, ..., p_{M-1},v_{M-1}_hat,a_{M-1}_hat, σ0..σ_{M-1}]
    MX z = MX::sym("z", n_interior*9 + M);

    // ── Unpack into knot list (M+1 knots, each [p;v;a] ∈ R⁹) ───────────────
    // Helper: convert Eigen→DM for boundary conditions
    auto e2dm = [](const Eigen::Vector3d& v) -> DM {
        return DM({v(0), v(1), v(2)});
    };

    // Boundary states (fixed)
    MX p0 = e2dm(start.p), v0 = e2dm(start.v), a0 = e2dm(start.a);
    MX pM = e2dm(goal.p),  vM = e2dm(goal.v),  aM = e2dm(goal.a);

    // Segment durations from σ
    MX T_sigma = z(Slice(n_interior*9, n_interior*9 + M));

    // Build symbolic T̄ᵢ (local average duration) for scaling
    auto T_bar = [&](int i) -> MX {
        if (i == 0) return MX::exp(T_sigma(0));
        if (i == M) return MX::exp(T_sigma(M-1));
        return 0.5*(MX::exp(T_sigma(i-1)) + MX::exp(T_sigma(i)));
    };

    // Build knot list: unscale derivative variables
    std::vector<MX> knots(M+1);
    // Knot 0 (fixed boundary)
    knots[0] = MX::vertcat({p0, v0, a0});
    // Interior knots i=1..M-1
    for (int i = 1; i <= M-1; ++i) {
        int idx = (i-1)*9;
        MX pi  = z(Slice(idx, idx+3));
        MX vi_hat = z(Slice(idx+3, idx+6));
        MX ai_hat = z(Slice(idx+6, idx+9));
        MX Tbar = T_bar(i);
        MX vi = vi_hat / Tbar;
        MX ai = ai_hat / (Tbar*Tbar);
        knots[i] = MX::vertcat({pi, vi, ai});
    }
    // Knot M (fixed boundary)
    knots[M] = MX::vertcat({pM, vM, aM});

    // ── Objective ────────────────────────────────────────────────────────────
    // Time cost
    MX J_time = MX::zeros(1);
    for (int s = 0; s < M; ++s)
        J_time += MX::exp(T_sigma(s));

    MX J = p_.w_time    * J_time
         + p_.w_smooth  * build_jerk_cost(knots, T_sigma)
         + p_.w_aero    * build_aero_cost(knots, T_sigma)
         + p_.w_bank    * build_bank_cost(knots, T_sigma)
         + p_.w_speed   * build_speed_cost(knots, T_sigma)
         + p_.w_acc     * build_acc_cost(knots, T_sigma);

    // ── Warm start ───────────────────────────────────────────────────────────
    // Recover scaled derivatives in warm-start
    auto z0_raw = warm_start(start, goal);

    // Rescale interior derivatives to scaled form  v̂ = T̄·v,  â = T̄²·a
    // (warm_start already returns scaled form; just copy to DM)
    std::vector<double> x0(z0_raw.begin(), z0_raw.end());
    DM x0_dm(x0);

    // ── NLP ──────────────────────────────────────────────────────────────────
    MXDict nlp = {{"x", z}, {"f", J}};

    Dict solver_opts;
    solver_opts["ipopt.max_iter"]  = p_.max_iter;
    solver_opts["ipopt.tol"]       = p_.tol;
    solver_opts["ipopt.print_level"] = 3;
    solver_opts["print_time"]      = false;

    Function solver = nlpsol("solver", "ipopt", nlp, solver_opts);
    DMDict sol = solver({{"x0", x0_dm}});

    bool success = static_cast<int>(solver.stats()["success"]) != 0;
    std::vector<double> z_opt = std::vector<double>(sol["x"]);

    // ── Unpack solution ───────────────────────────────────────────────────────
    FWTrajectory traj;
    traj.success = success;
    traj.msg     = success ? "OK" : "IPOPT failed";
    traj.cost    = static_cast<double>(sol["f"]);

    traj.knots.resize(M+1);
    // Boundary knots
    traj.knots[0] = start;
    traj.knots[M] = goal;

    for (int i = 1; i <= M-1; ++i) {
        int idx = (i-1)*9;
        KnotState ks;
        ks.p = Eigen::Vector3d(z_opt[idx], z_opt[idx+1], z_opt[idx+2]);
        // Recover un-scaled velocity / accel
        // T̄ᵢ from optimised σ
        double Ts_prev = std::exp(z_opt[n_interior*9 + i - 1]);
        double Ts_next = (i < M) ? std::exp(z_opt[n_interior*9 + i]) : Ts_prev;
        double Tbar = (i == 0) ? Ts_prev :
                      (i == M) ? Ts_next : 0.5*(Ts_prev + Ts_next);
        ks.v = Eigen::Vector3d(z_opt[idx+3], z_opt[idx+4], z_opt[idx+5]) / Tbar;
        ks.a = Eigen::Vector3d(z_opt[idx+6], z_opt[idx+7], z_opt[idx+8]) / (Tbar*Tbar);
        traj.knots[i] = ks;
    }

    traj.T_seg.resize(M);
    for (int s = 0; s < M; ++s)
        traj.T_seg[s] = std::exp(z_opt[n_interior*9 + s]);

    return traj;
}

// ─────────────────────────────────────────────────────────────────────────────
//  FWTrajectory::position / velocity  (evaluation at arbitrary time)
// ─────────────────────────────────────────────────────────────────────────────
Eigen::Vector3d FWTrajectory::position(double t) const {
    double cum = 0.0;
    int M = static_cast<int>(T_seg.size());
    for (int s = 0; s < M; ++s) {
        if (t <= cum + T_seg[s] || s == M-1) {
            double tau = (t - cum) / T_seg[s];
            tau = std::max(0.0, std::min(1.0, tau));
            return hermite::eval_pos(knots[s], knots[s+1], T_seg[s], tau);
        }
        cum += T_seg[s];
    }
    return knots[M].p;
}

Eigen::Vector3d FWTrajectory::velocity(double t) const {
    double cum = 0.0;
    int M = static_cast<int>(T_seg.size());
    for (int s = 0; s < M; ++s) {
        if (t <= cum + T_seg[s] || s == M-1) {
            double tau = (t - cum) / T_seg[s];
            tau = std::max(0.0, std::min(1.0, tau));
            return hermite::eval_vel(knots[s], knots[s+1], T_seg[s], tau);
        }
        cum += T_seg[s];
    }
    return knots[M].v;
}

} // namespace fw_hermite
