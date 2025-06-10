import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import multiprocessing as mp

# System parameters
delta = 0.1
omega0 = 1.0
gamma = -0.2 # softening

# Frequency sweep
frequencies = np.linspace(0.5, 1.5, 100)

# Time integration
t_end = 300
t_eval = np.linspace(0, t_end, 5000)
x0 = [0.0, 0.0]

# Forcing amplitudes to test in parallel
forcing_amplitudes = [0.01, 0.02, 0.1, 0.2, 0.3]

def softening_oscillator(t, y, delta, omega0, gamma, F, omega):
    x, dx = y
    ddx = F * np.cos(omega * t) - delta * dx - omega0**2 * x - gamma * x**3
    return [dx, ddx]

def get_steady_amplitude(t, x):
    x_steady = x[int(0.8 * len(x)):]
    peaks, _ = find_peaks(x_steady)
    if len(peaks) >= 2:
        return (np.max(x_steady[peaks]) - np.min(x_steady)) / 2
    else:
        return np.max(np.abs(x_steady))

def compute_frf(F):
    amplitudes = []
    for omega in frequencies:
        sol = solve_ivp(
            softening_oscillator,
            [0, t_end],
            x0,
            t_eval=t_eval,
            args=(delta, omega0, gamma, F, omega),
            rtol=1e-8,
            atol=1e-10
        )
        amp = get_steady_amplitude(sol.t, sol.y[0])
        amplitudes.append(amp)
    return F, amplitudes

def main():
    with mp.Pool(processes=min(len(forcing_amplitudes), mp.cpu_count())) as pool:
        results = pool.map(compute_frf, forcing_amplitudes)

    # Sort results by forcing amplitude
    results.sort(key=lambda x: x[0])

    # Plot FRFs
    plt.figure(figsize=(10, 6))
    for F, amps in results:
        plt.plot(frequencies, amps, label=f"F = {F}")
    plt.xlabel("Excitation Frequency ω [rad/s]")
    plt.ylabel("Amplitude")
    plt.title("FRF of Softening Oscillator")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
