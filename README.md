# Molmer-Sorensen-gate
The code is for people (like myself) to understand how the famous two-qubit gate evolves.
The purpose of the notebook is for my lab (https://www.quantumlah.org/research/group/Dzmitry) to fit the experiment result and fine tune the experiment parameters due to hardware or software limitations to achieve better gate performance.

For the original proposal of this quantum gate, please refer to the original paper (Phys.Rev.Lett. 82 (1999) 1971-1974). For our simulation and experiment reference in trapped ion systems, please check this one (10.1103/PhysRevA.72.062316).

Building a quantum computer is quite fun lol.

## Contents
- `ms_gate.py`: the model and the scans (QuTiP 5). The docstring at the top states the Hamiltonian, the units and the noise model.
- `MS gate simulating Delay scan, Parity and Frequency.ipynb`: notebook that runs the scans and plots them.
- `test_ms_gate.py`: checks against closed-form results.

## What it simulates
Two ions and one motional mode driven on the red and blue sidebands, starting from `|ee>` and a thermal motional state.

| scan | what is varied | what you get |
|---|---|---|
| `delay_scan` | pulse duration | brightness, populations, Bell-state fidelity |
| `frequency_scan` | symmetric detuning, with optional frequency offsets | brightness, fidelity |
| `parity_scan` | phase of a pi/2 analysis pulse after the gate | parity fringe, contrast, fidelity |

Decoherence is included through a Lindblad master equation: motional heating, motional dephasing and qubit dephasing.

Not included: spectator modes, off-resonant carrier coupling, AC Stark shifts, higher orders in the Lamb-Dicke parameter, pulse shaping, spontaneous scattering, and errors of the analysis pulse.

## Units
Time is in sideband pi-times, detunings are in units of the sideband Rabi frequency, and noise rates are per pi-time. The ideal gate is `bsb=2, rsb=-2, t=1`.

## Usage
```bash
pip install -r requirements.txt
```
```python
import ms_gate as ms

scan = ms.parity_scan(t_gate=1, bsb=2, rsb=-2, nbar=0.3,
                      heating_rate=0.05, qubit_dephasing=0.03)
print(scan["contrast"], scan["fidelity"])
ms.plot_parity_scan(scan)
```
```bash
pytest
```
