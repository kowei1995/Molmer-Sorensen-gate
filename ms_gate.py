"""Molmer-Sorensen gate simulation for two trapped ions sharing one motional mode.

Written for QuTiP 5.

Model
-----
Interaction picture, Lamb-Dicke regime to first order, rotating-wave
approximation. A bichromatic field drives the blue and the red sideband of
both ions:

    H(t) = sum_j (g_j / 2) * [ s+_j c^dag exp(-i d_b t)
                             + s+_j c     exp(-i d_r t) + h.c. ]

    s+_j : raising operator of ion j (|g> -> |e>)
    c    : annihilation operator of the motional mode
    g_j  : sideband Rabi frequency (eta * Omega) of ion j
    d_b  : detuning of the blue tone from the blue sideband
    d_r  : detuning of the red tone from the red sideband

Units
-----
Time is measured in sideband pi-times, t_pi = pi / g, so g = pi internally.
Detunings are given in units of g. Decoherence rates are per pi-time.

With the symmetric choice d_b = -d_r = d the propagator is exactly

    U(t) = D(alpha(t) S_x) exp(i Phi(t) S_x^2),   S_x = sx_1 + sx_2
    Phi(t) = (g/2)^2 * [ t/d - sin(d t)/d^2 ]

The phase-space loop closes at t = 2 pi / d, and the gate is maximally
entangling when d = 2 g. In the units used here that is bsb=2, rsb=-2, t=1.

Decoherence (Lindblad)
----------------------
heating_rate      : motional heating, d<n>/dt in quanta per pi-time.
                    Collapse operators sqrt(r) c and sqrt(r) c^dag.
motional_dephasing: collapse operator sqrt(r) c^dag c. The coherence between
                    Fock states n and m decays at r (n - m)^2 / 2.
qubit_dephasing   : collapse operators sqrt(r / 2) sz_j, independent on each
                    ion. A single-qubit coherence decays at rate r.

All three operators keep their form in the interaction picture, so they can
be used together with H(t) above without modification.
"""

from __future__ import annotations

import numpy as np
import qutip as qt

G = np.pi  # sideband Rabi frequency, so that one time unit is one pi-time

# two-qubit basis ordering used by QuTiP for tensor(ion1, ion2), 0 = g, 1 = e
GG, GE, EG, EE = 0, 1, 2, 3


# --------------------------------------------------------------------------
# system
# --------------------------------------------------------------------------
def _operators(n_fock):
    sm = qt.destroy(2)  # |e> -> |g>
    eye2, eye_c = qt.qeye(2), qt.qeye(n_fock)
    sm1 = qt.tensor(sm, eye2, eye_c)
    sm2 = qt.tensor(eye2, sm, eye_c)
    sz1 = qt.tensor(qt.sigmaz(), eye2, eye_c)
    sz2 = qt.tensor(eye2, qt.sigmaz(), eye_c)
    c = qt.tensor(eye2, eye2, qt.destroy(n_fock))
    return sm1, sm2, sz1, sz2, c


def hamiltonian(bsb, rsb, n_fock, coupling=(1.0, 1.0)):
    """Time-dependent MS Hamiltonian as a QobjEvo.

    bsb, rsb : detunings of the two tones from their sidebands, in units of g.
    coupling : relative sideband Rabi frequency of each ion (1 = nominal).
    """
    sm1, sm2, _, _, c = _operators(n_fock)
    d_b, d_r = bsb * G, rsb * G
    s_plus = 0.5 * G * (coupling[0] * sm1.dag() + coupling[1] * sm2.dag())
    blue = s_plus * c.dag()
    red = s_plus * c
    return qt.QobjEvo([
        [blue, lambda t: np.exp(-1j * d_b * t)],
        [blue.dag(), lambda t: np.exp(1j * d_b * t)],
        [red, lambda t: np.exp(-1j * d_r * t)],
        [red.dag(), lambda t: np.exp(1j * d_r * t)],
    ])


def collapse_operators(n_fock, heating_rate=0.0, motional_dephasing=0.0,
                       qubit_dephasing=0.0):
    _, _, sz1, sz2, c = _operators(n_fock)
    c_ops = []
    if heating_rate > 0:
        c_ops += [np.sqrt(heating_rate) * c, np.sqrt(heating_rate) * c.dag()]
    if motional_dephasing > 0:
        c_ops.append(np.sqrt(motional_dephasing) * c.dag() * c)
    if qubit_dephasing > 0:
        c_ops += [np.sqrt(qubit_dephasing / 2) * sz1,
                  np.sqrt(qubit_dephasing / 2) * sz2]
    return c_ops


def evolve(tlist, bsb=2.0, rsb=-2.0, nbar=0.0, n_fock=15, coupling=(1.0, 1.0),
           heating_rate=0.0, motional_dephasing=0.0, qubit_dephasing=0.0):
    """Evolve |ee> x thermal(nbar) and return the two-qubit density matrix
    (4x4 numpy array, motion traced out) at every time in tlist."""
    tlist = np.atleast_1d(np.asarray(tlist, dtype=float))
    rho0 = qt.tensor(qt.fock_dm(2, 1), qt.fock_dm(2, 1),
                     qt.thermal_dm(n_fock, nbar))
    # mesolve needs tlist to start at the initial time
    times = tlist if tlist[0] == 0 else np.concatenate([[0.0], tlist])
    if len(times) == 1:
        return [rho0.ptrace([0, 1]).full()]
    result = qt.mesolve(
        hamiltonian(bsb, rsb, n_fock, coupling), rho0, times,
        c_ops=collapse_operators(n_fock, heating_rate, motional_dephasing,
                                 qubit_dephasing),
        options={"nsteps": 1_000_000, "store_states": True,
                 "progress_bar": False},
    )
    states = result.states if tlist[0] == 0 else result.states[1:]
    return [s.ptrace([0, 1]).full() for s in states]


# --------------------------------------------------------------------------
# observables on the 4x4 two-qubit density matrix
# --------------------------------------------------------------------------
def populations(rho):
    """(P_gg, P_ge, P_eg, P_ee)."""
    return np.real(np.diag(rho))


def brightness(rho):
    """Mean number of ions in |g> (the bright state): 2 P_gg + P_ge + P_eg."""
    p = populations(rho)
    return 2 * p[GG] + p[GE] + p[EG]


def population_parity(rho):
    """P_gg + P_ee - P_ge - P_eg. Uses populations only: it does not tell a
    Bell state from an incoherent mixture of |gg> and |ee>."""
    p = populations(rho)
    return p[GG] + p[EE] - p[GE] - p[EG]


def bell_fidelity(rho):
    """Fidelity with the closest state (|gg> + exp(i phi)|ee>)/sqrt(2):
    F = (P_gg + P_ee)/2 + |rho_{gg,ee}|."""
    p = populations(rho)
    return 0.5 * (p[GG] + p[EE]) + abs(rho[GG, EE])


def analysis_pulse(rho, phase):
    """Apply a global pi/2 pulse with laser phase `phase` to both ions."""
    axis = np.cos(phase) * qt.sigmax() + np.sin(phase) * qt.sigmay()
    r = (-1j * np.pi / 4 * axis).expm()
    u = qt.tensor(r, r).full()
    return u @ rho @ u.conj().T


# --------------------------------------------------------------------------
# scans
# --------------------------------------------------------------------------
def delay_scan(t_start=0.0, t_end=5.0, points_per_pi_time=50, **kwargs):
    """Fixed detuning, scan the pulse duration.

    Returns a dict with arrays: t, brightness, populations (n, 4),
    population_parity, fidelity. kwargs go to `evolve`."""
    n = max(int(round(abs(t_end - t_start) * points_per_pi_time)) + 1, 2)
    t = np.linspace(t_start, t_end, n)
    rhos = evolve(t, **kwargs)
    return {
        "t": t,
        "brightness": np.array([brightness(r) for r in rhos]),
        "populations": np.array([populations(r) for r in rhos]),
        "population_parity": np.array([population_parity(r) for r in rhos]),
        "fidelity": np.array([bell_fidelity(r) for r in rhos]),
    }


def frequency_scan(detunings=None, t=1.0, bsb_offset=0.0, rsb_offset=0.0,
                   **kwargs):
    """Fixed pulse duration t, scan the symmetric detuning d
    (blue tone at +d, red tone at -d).

    bsb_offset / rsb_offset model how far the frequencies typed in the
    experiment are from the true sideband frequencies."""
    if detunings is None:
        detunings = np.linspace(-5, 5, 101)
    detunings = np.asarray(detunings, dtype=float)
    rhos = [evolve([t], bsb=bsb_offset + d, rsb=rsb_offset - d, **kwargs)[0]
            for d in detunings]
    return {
        "detuning": detunings,
        "brightness": np.array([brightness(r) for r in rhos]),
        "populations": np.array([populations(r) for r in rhos]),
        "fidelity": np.array([bell_fidelity(r) for r in rhos]),
    }


def parity_scan(t_gate=1.0, n_phases=73, **kwargs):
    """Run the MS pulse for t_gate, then apply a global pi/2 analysis pulse
    with phase phi and measure the parity <sz sz>. Scanning phi gives

        parity(phi) = offset + contrast * cos(2 phi + phi0)

    with contrast = 2 |rho_{gg,ee}|, so the measured Bell-state fidelity is
    F = (P_gg + P_ee)/2 + contrast/2.

    The analysis pulse is ideal and instantaneous. kwargs go to `evolve`."""
    rho = evolve([t_gate], **kwargs)[0]
    phases = np.linspace(0, 2 * np.pi, n_phases)
    parity = np.array([population_parity(analysis_pulse(rho, p))
                       for p in phases])

    # linear least squares for offset + a cos(2 phi) + b sin(2 phi)
    design = np.column_stack([np.ones_like(phases), np.cos(2 * phases),
                              np.sin(2 * phases)])
    offset, a, b = np.linalg.lstsq(design, parity, rcond=None)[0]
    contrast = float(np.hypot(a, b))
    pops = populations(rho)
    return {
        "phase": phases,
        "parity": parity,
        "fit": design @ np.array([offset, a, b]),
        "contrast": contrast,
        "offset": float(offset),
        "populations": pops,
        "fidelity": float(0.5 * (pops[GG] + pops[EE]) + 0.5 * contrast),
        "rho": rho,
    }


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
def plot_delay_scan(scan, title="MS gate delay scan"):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    t = scan["t"]
    axes[0].plot(t, scan["brightness"])
    axes[0].set_ylabel("brightness  2P_gg + P_ge + P_eg")
    axes[0].set_title(title)
    for idx, label in ((GG, "P_gg"), (EE, "P_ee")):
        axes[1].plot(t, scan["populations"][:, idx], label=label)
    axes[1].plot(t, scan["populations"][:, GE] + scan["populations"][:, EG],
                 label="P_ge + P_eg")
    axes[1].set_ylabel("populations")
    axes[1].legend()
    axes[2].plot(t, scan["fidelity"], label="Bell-state fidelity")
    axes[2].plot(t, scan["population_parity"], "--",
                 label="population parity")
    axes[2].set_ylabel("fidelity / parity")
    axes[2].set_xlabel("pulse duration (sideband pi-times)")
    axes[2].legend()
    for ax in axes:
        ax.grid(True)
    fig.tight_layout()
    return fig


def plot_frequency_scan(scan, title="MS gate frequency scan"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(scan["detuning"], scan["brightness"], label="brightness")
    ax.plot(scan["detuning"], scan["fidelity"], label="Bell-state fidelity")
    ax.set_xlabel("symmetric detuning (units of sideband Rabi frequency)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_parity_scan(scan, title="MS gate parity scan"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(scan["phase"] / np.pi, scan["parity"], "o", ms=4, label="parity")
    ax.plot(scan["phase"] / np.pi, scan["fit"], label="fit")
    ax.set_xlabel("analysis pulse phase (units of pi)")
    ax.set_ylabel("parity  P_gg + P_ee - P_ge - P_eg")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(f"{title}: contrast = {scan['contrast']:.4f}, "
                 f"fidelity = {scan['fidelity']:.4f}")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig
