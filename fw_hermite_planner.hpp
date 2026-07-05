/**
 * fw_hermite_planner.hpp
 *
 * Fixed-Wing Hermite Spline Trajectory Optimizer  —  MIGHTY-style
 * ----------------------------------------------------------------
 * Adapts the MIGHTY quadrotor framework (Kondo et al. 2025, arXiv:2511.10822)
 * to a fixed-wing UAV.  Instead of quadrotor thrust / attitude costs the
 * objective penalises:
 *   - Integrated jerk (smoothness)
 *   - Time (encourage short missions)
 *   - Lift / drag residual  (simplified point-mass aerodynamics)
 *   - Bank-angle limit violations  (soft hinge)
 *   - Speed limits  (stall + VNE)
 *   - Turn-rate / load-factor limits
 *
 * Representation
 * --------------
 * Quintic Hermite spline (degree 5, ν=2): each knot stores p, v, a ∈ R³.
 * Segment s ∈ {0 … M-1} maps normalised time τ ∈ [0,1] via the six
 * standard quintic Hermite basis functions h₀…h₅.
 *
 * The Hermite→Bézier affine map (eq. 2 in the paper) is used to evaluate
 * jerk-integral costs in closed form and sample-based aerodynamic costs
 * via the Bernstein basis.
 *
 * Optimisation
 * ------------
 * Unconstrained NLP via CasADi + IPOPT (or L-BFGS-B).
 * Duration diffeomorphism:  Ts = exp(σs),  σs ∈ ℝ  (always positive).
 * Scaled derivative variables:  v̂ᵢ = T̄ᵢ vᵢ,  âᵢ = T̄ᵢ² aᵢ.
 *
 * Warm start
 * ----------
 * A Dubins path (constant-speed arc-line-arc) provides the initial
 * waypoint sequence; durations are initialised from the arc lengths.
 *
 * Dependencies
 * ------------
 *   Eigen3   (linear algebra)
 *   CasADi   (symbolic AD + IPOPT interface)
 *   yaml-cpp (optional, for parameter loading)
 *
 * Build example (CMake snippet):
 *   find_package(casadi REQUIRED)
 *   find_package(Eigen3 REQUIRED)
 *   target_link_libraries(my_node casadi Eigen3::Eigen)
 */

#pragma once

#include <Eigen/Dense>
#include <casadi/casadi.hpp>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace fw_hermite {

// ─────────────────────────────────────────────────────────────────────────────
//  Planner parameters
// ─────────────────────────────────────────────────────────────────────────────
struct FWParams {
    // ── Spline ──────────────────────────────────────────────────────────────
    int    M          = 6;        ///< Number of segments
    int    kappa      = 8;        ///< Samples per segment for integral costs

    // ── Weights ─────────────────────────────────────────────────────────────
    double w_time     = 5e1;      ///< Time penalty  (encourages short missions)
    double w_smooth   = 1e-1;     ///< Jerk smoothness (closed-form integral)
    double w_aero     = 1e2;      ///< Lift/drag residual
    double w_bank     = 1e3;      ///< Bank-angle soft hinge
    double w_speed    = 1e3;      ///< Speed limit soft hinge
    double w_acc      = 5e2;      ///< Acceleration magnitude soft hinge

    // ── Fixed-wing aerodynamics (point-mass) ────────────────────────────────
    double mass       = 2.5;      ///< kg
    double g          = 9.81;     ///< m/s²
    double rho        = 1.225;    ///< kg/m³  (ISA sea level)
    double S_wing     = 0.35;     ///< m²  reference wing area
    double CL0        = 0.3;      ///< Lift coefficient at zero AoA
    double CLa        = 5.0;      ///< dCL/dα  [1/rad]
    double CD0        = 0.025;    ///< Parasitic drag coefficient
    double k_induced  = 0.045;    ///< Induced drag factor  (k·CL²)
    double V_stall    = 8.0;      ///< m/s  stall speed
    double V_ne       = 30.0;     ///< m/s  never-exceed speed

    // ── Kinematic / structural limits ───────────────────────────────────────
    double phi_max    = 60.0 * M_PI / 180.0;  ///< Max bank angle [rad]
    double a_max      = 15.0;     ///< m/s²  max total acceleration magnitude
    double n_max      = 3.5;      ///< max load factor (structural)

    // ── Optimiser ───────────────────────────────────────────────────────────
    int    max_iter   = 500;
    double tol        = 1e-5;
    std::string solver = "ipopt"; ///< "ipopt" or "lbfgs" (CasADi plugins)
};

// ─────────────────────────────────────────────────────────────────────────────
//  State at a single knot
// ─────────────────────────────────────────────────────────────────────────────
struct KnotState {
    Eigen::Vector3d p;   ///< Position    [m]
    Eigen::Vector3d v;   ///< Velocity    [m/s]
    Eigen::Vector3d a;   ///< Acceleration [m/s²]
};

// ─────────────────────────────────────────────────────────────────────────────
//  Result trajectory
// ─────────────────────────────────────────────────────────────────────────────
struct FWTrajectory {
    std::vector<KnotState>    knots;    ///< M+1 knot states
    std::vector<double>       T_seg;    ///< M segment durations [s]
    double                    cost;
    bool                      success;
    std::string               msg;

    // Convenience: sample position at global time t  (binary search + Hermite)
    Eigen::Vector3d position(double t) const;
    Eigen::Vector3d velocity(double t) const;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Quintic Hermite basis functions  h₀…h₅  on τ ∈ [0,1]
// ─────────────────────────────────────────────────────────────────────────────
namespace hermite {

/**
 * Evaluate the 6 quintic Hermite basis functions at τ.
 * Ordering matches MIGHTY eq. (1):
 *   h₀(τ) … h₅(τ)
 * Returns a 6-vector.
 */
inline Eigen::Matrix<double,6,1> basis(double tau) {
    const double t  = tau;
    const double t2 = t*t, t3 = t2*t, t4 = t3*t, t5 = t4*t;
    Eigen::Matrix<double,6,1> h;
    h(0) =  1 - 10*t3 + 15*t4 -  6*t5;
    h(1) =  t  - 6*t3 +  8*t4 -  3*t5;
    h(2) =  0.5*t2 - 1.5*t3 + 1.5*t4 - 0.5*t5;
    h(3) =  10*t3 - 15*t4 +  6*t5;
    h(4) = -4*t3  +  7*t4 -  3*t5;
    h(5) =  0.5*t3 - t4   + 0.5*t5;
    return h;
}

/** First derivative dh/dτ */
inline Eigen::Matrix<double,6,1> dbasis(double tau) {
    const double t  = tau;
    const double t2 = t*t, t3 = t2*t, t4 = t3*t;
    Eigen::Matrix<double,6,1> dh;
    dh(0) = -30*t2 + 60*t3 - 30*t4;
    dh(1) =  1 - 18*t2 + 32*t3 - 15*t4;
    dh(2) =  t  -  4.5*t2 +  6*t3 - 2.5*t4;
    dh(3) =  30*t2 - 60*t3 + 30*t4;
    dh(4) = -12*t2 + 28*t3 - 15*t4;
    dh(5) =   1.5*t2 -  4*t3 + 2.5*t4;
    return dh;
}

/** Second derivative d²h/dτ² */
inline Eigen::Matrix<double,6,1> ddbasis(double tau) {
    const double t  = tau;
    const double t2 = t*t, t3 = t2*t;
    Eigen::Matrix<double,6,1> ddh;
    ddh(0) = -60*t  + 180*t2 - 120*t3;
    ddh(1) = -36*t  +  96*t2 -  60*t3;
    ddh(2) =  1 -   9*t  +  18*t2 -  10*t3;
    ddh(3) =  60*t  - 180*t2 + 120*t3;
    ddh(4) = -24*t  +  84*t2 -  60*t3;
    ddh(5) =   3*t  -  12*t2 +  10*t3;
    return ddh;
}

/** Evaluate position on segment s at normalised τ */
inline Eigen::Vector3d eval_pos(const KnotState& ks, const KnotState& ke,
                                 double Ts, double tau) {
    auto h = basis(tau);
    return h(0)*ks.p + h(1)*(Ts*ks.v) + h(2)*(Ts*Ts*ks.a*0.5)
          +h(3)*ke.p + h(4)*(Ts*ke.v) + h(5)*(Ts*Ts*ke.a*0.5);
}

/** Evaluate velocity  dx/dt = (1/Ts) * dx/dτ */
inline Eigen::Vector3d eval_vel(const KnotState& ks, const KnotState& ke,
                                 double Ts, double tau) {
    auto dh = dbasis(tau);
    Eigen::Vector3d dxdtau =
        dh(0)*ks.p + dh(1)*(Ts*ks.v) + dh(2)*(Ts*Ts*ks.a*0.5)
       +dh(3)*ke.p + dh(4)*(Ts*ke.v) + dh(5)*(Ts*Ts*ke.a*0.5);
    return dxdtau / Ts;
}

/** Evaluate acceleration  d²x/dt² */
inline Eigen::Vector3d eval_acc(const KnotState& ks, const KnotState& ke,
                                 double Ts, double tau) {
    auto ddh = ddbasis(tau);
    Eigen::Vector3d d2xdtau =
        ddh(0)*ks.p + ddh(1)*(Ts*ks.v) + ddh(2)*(Ts*Ts*ks.a*0.5)
       +ddh(3)*ke.p + ddh(4)*(Ts*ke.v) + ddh(5)*(Ts*Ts*ke.a*0.5);
    return d2xdtau / (Ts*Ts);
}

} // namespace hermite


// ─────────────────────────────────────────────────────────────────────────────
//  Dubins warm-start
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Generate M+1 waypoints along the 2-D Dubins path (RSR / LSL etc.)
 * from start to goal (ignoring altitude for the path geometry; altitude
 * is linearly interpolated).
 *
 * Returns equally-spaced knot positions along the Dubins path arc.
 */
std::vector<Eigen::Vector3d> dubins_waypoints(
    const KnotState& start,
    const KnotState& goal,
    int M,
    double turn_radius = 20.0);

// ─────────────────────────────────────────────────────────────────────────────
//  Main planner class
// ─────────────────────────────────────────────────────────────────────────────
class FWHermitePlanner {
public:
    explicit FWHermitePlanner(const FWParams& params = FWParams{});

    /**
     * Plan a trajectory from `start` to `goal`.
     * Both boundary states (position, velocity, acceleration) are fixed.
     * Interior knot states and per-segment durations are the decision vars.
     */
    FWTrajectory plan(const KnotState& start, const KnotState& goal);

private:
    FWParams p_;

    // ── CasADi symbolic problem builders ────────────────────────────────────
    casadi::MX build_jerk_cost(const std::vector<casadi::MX>& knots,
                               const casadi::MX& T_sigma) const;

    casadi::MX build_aero_cost(const std::vector<casadi::MX>& knots,
                               const casadi::MX& T_sigma) const;

    casadi::MX build_bank_cost(const std::vector<casadi::MX>& knots,
                               const casadi::MX& T_sigma) const;

    casadi::MX build_speed_cost(const std::vector<casadi::MX>& knots,
                                const casadi::MX& T_sigma) const;

    casadi::MX build_acc_cost(const std::vector<casadi::MX>& knots,
                              const casadi::MX& T_sigma) const;

    // ── Warm-start initialisation ────────────────────────────────────────────
    std::vector<double> warm_start(const KnotState& start,
                                   const KnotState& goal) const;

    // ── Helpers ──────────────────────────────────────────────────────────────
    /** Hermite→Bézier control points for segment s (symbolic) */
    std::vector<casadi::MX> hermite_to_bezier(
        const casadi::MX& ps, const casadi::MX& vs, const casadi::MX& as,
        const casadi::MX& pe, const casadi::MX& ve, const casadi::MX& ae,
        const casadi::MX& Ts) const;

    /** Closed-form integrated squared-jerk on one Bézier segment (symbolic) */
    casadi::MX bezier_jerk_integral(const std::vector<casadi::MX>& ctrl,
                                    const casadi::MX& Ts) const;

    /** Smooth hinge  ϕ(x) = (1/μ) log(1 + exp(μ·x)),  μ = 50 */
    casadi::MX smooth_hinge(const casadi::MX& x) const;

    /** Bank angle from centripetal acceleration vector and gravity */
    casadi::MX bank_angle_sym(const casadi::MX& vel,
                              const casadi::MX& acc) const;
};

} // namespace fw_hermite
