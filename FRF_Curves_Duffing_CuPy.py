import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import multiprocessing as mp
from numba import njit

# Duffing oscillator parameters
delta = 1
alpha = 1
beta = 1

# Frequency sweep
frequencies = np.linspace(0.001, 2*np.pi, round((2*np.pi - 0.001) / (1e-3)))
t_end = 100
t_eval = np.linspace(0, t_end, 5000)
x0 = [0.0, 0.0]

# Forcing amplitudes to sweep in parallel
forcing_amplitudes = np.linspace(1, 37, 18)

@njit
def duffing_rhs(t, y, delta, alpha, beta, omega, F):
    x, dx = y
    ddx = F * np.cos(omega * t) - delta * dx - alpha * x - beta * x**3
    return np.array([dx, ddx])

# Wrapper for solve_ivp, which can't use Numba functions directly
def duffing(t, y, delta, alpha, beta, omega, F):
    return duffing_rhs(t, y, delta, alpha, beta, omega, F)

@njit
def get_peak_amplitude(signal):
    # Simple peak-to-peak approximation since find_peaks can't be JITed
    n = len(signal)
    max_val = -1e9
    min_val = 1e9
    for i in range(n):
        if signal[i] > max_val:
            max_val = signal[i]
        if signal[i] < min_val:
            min_val = signal[i]
    return 0.5 * (max_val - min_val)

def get_steady_amplitude(t, x):
    steady_x = x[int(0.9 * len(x)):]
    return get_peak_amplitude(steady_x)

def compute_frf_for_F(F):
    amps = []
    for omega in frequencies:
        sol = solve_ivp(duffing, [0, t_end], x0, t_eval=t_eval,
                        args=(delta, alpha, beta, omega, F),
                        rtol=1e-6, atol=1e-8)
        amp = get_steady_amplitude(sol.t, sol.y[0])
        amps.append(amp)
    return F, amps

def main():
    with mp.Pool(processes=min(len(forcing_amplitudes), mp.cpu_count())) as pool:
        results = pool.map(compute_frf_for_F, forcing_amplitudes)

    results.sort(key=lambda x: x[0])  # Sort by forcing amplitude

    # Save to CSV
    with open("duffing_frf_data.csv", "w") as f:
        header = "omega," + ",".join([f"F={F:.2f}" for F, _ in results]) + "\n"
        f.write(header)
        for i in range(len(frequencies)):
            row = [f"{frequencies[i]:.6f}"] + [f"{amps[i]:.6f}" for _, amps in results]
            f.write(",".join(row) + "\n")

    # Plot
    plt.figure(figsize=(10, 6))
    for F, amps in results:
        plt.plot(frequencies, amps, label=f"F = {F:.2f}")
    plt.xlabel("Excitation Frequency ω [rad/s]")
    plt.ylabel("Amplitude")
    plt.title("Duffing FRF with Numba-accelerated Components")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
