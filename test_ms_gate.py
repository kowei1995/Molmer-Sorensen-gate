"""Checks of ms_gate against closed-form results. Run with `pytest`."""

import numpy as np
import pytest

import ms_gate as ms


def test_ideal_gate_makes_bell_state():
    rho = ms.evolve([1.0], bsb=2, rsb=-2)[0]
    p = ms.populations(rho)
    assert p[ms.GG] == pytest.approx(0.5, abs=1e-5)
    assert p[ms.EE] == pytest.approx(0.5, abs=1e-5)
    assert ms.bell_fidelity(rho) == pytest.approx(1.0, abs=1e-5)


def test_gate_is_insensitive_to_temperature():
    rho = ms.evolve([1.0], nbar=0.3, n_fock=25)[0]
    assert ms.bell_fidelity(rho) == pytest.approx(1.0, abs=1e-4)


def test_two_pi_times_flips_both_ions():
    rho = ms.evolve([2.0])[0]
    assert ms.brightness(rho) == pytest.approx(2.0, abs=1e-5)


def test_two_loops_at_double_detuning():
    # d = 4g, t = 1: two closed loops, Phi = pi/16, P_gg = sin^2(pi/8)
    rho = ms.evolve([1.0], bsb=4, rsb=-4)[0]
    assert ms.brightness(rho) == pytest.approx(2 * np.sin(np.pi / 8) ** 2,
                                               abs=1e-5)


def test_parity_scan_matches_density_matrix():
    scan = ms.parity_scan(qubit_dephasing=0.05, heating_rate=0.05)
    assert scan["contrast"] == pytest.approx(
        2 * abs(scan["rho"][ms.GG, ms.EE]), abs=1e-8)
    assert scan["fidelity"] == pytest.approx(ms.bell_fidelity(scan["rho"]),
                                             abs=1e-8)
    assert np.allclose(scan["parity"], scan["fit"], atol=1e-8)


def test_parity_scan_separates_mixture_from_bell_state():
    mixture = np.diag([0.5, 0, 0, 0.5]).astype(complex)
    assert ms.population_parity(mixture) == pytest.approx(1.0)
    phases = np.linspace(0, 2 * np.pi, 9)
    parity = [ms.population_parity(ms.analysis_pulse(mixture, p))
              for p in phases]
    assert np.allclose(parity, 0, atol=1e-12)


def test_qubit_dephasing_rate():
    # no drive: a single-qubit coherence must decay as exp(-rate * t)
    import qutip as qt
    rate, t = 0.3, 2.0
    plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    rho0 = qt.tensor(qt.ket2dm(plus), qt.fock_dm(2, 0), qt.fock_dm(3, 0))
    h0 = 0 * qt.tensor(qt.qeye(2), qt.qeye(2), qt.qeye(3))
    out = qt.mesolve(h0, rho0, [0, t],
                     c_ops=ms.collapse_operators(3, qubit_dephasing=rate))
    coherence = out.states[-1].ptrace(0).full()[0, 1]
    assert abs(coherence) == pytest.approx(0.5 * np.exp(-rate * t), rel=1e-4)


def test_heating_rate():
    # no drive: <n> must grow linearly at the heating rate
    import qutip as qt
    rate, t, n_fock = 0.2, 2.0, 30
    rho0 = qt.tensor(qt.fock_dm(2, 0), qt.fock_dm(2, 0), qt.fock_dm(n_fock, 0))
    num = qt.tensor(qt.qeye(2), qt.qeye(2), qt.num(n_fock))
    out = qt.mesolve(0 * num, rho0, [0, t],
                     c_ops=ms.collapse_operators(n_fock, heating_rate=rate),
                     e_ops=[num])
    assert out.expect[0][-1] == pytest.approx(rate * t, rel=1e-3)


@pytest.mark.parametrize("noise", ["heating_rate", "motional_dephasing",
                                   "qubit_dephasing"])
def test_decoherence_lowers_fidelity(noise):
    rho = ms.evolve([1.0], **{noise: 0.05})[0]
    assert 0.5 < ms.bell_fidelity(rho) < 0.999
