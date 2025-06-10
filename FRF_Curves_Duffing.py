import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import multiprocessing as mp
# from numba import njit


# Duffing oscillator parameters
delta = 1
alpha = 1
beta = 1

# Frequency sweep
frequencies = np.linspace(0.001, 2*np.pi, round((2*np.pi-0.001)/(1000**-1)))
t_end = 100
t_eval = np.linspace(0, t_end, 5000)
x0 = [0.0, 0.0]

# Forcing amplitudes to sweep in parallel
forcing_amplitudes = np.linspace(1,37,19)

def duffing(t, y, delta, alpha, beta, omega, F):
    x, dx = y
    ddx = F * np.cos(omega * t) - delta * dx - alpha * x - beta * x**3
    return [dx, ddx]

def get_steady_amplitude(t, x):
    steady_x = x[int(0.9 * len(x)):]
    peaks, _ = find_peaks(steady_x)
    if len(peaks) >= 2:
        amp = (np.max(steady_x[peaks]) - np.min(steady_x)) / 2
    else:
        amp = np.max(np.abs(steady_x))
    return amp

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
    plt.title("Duffing FRF at Different Forcing Amplitudes")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
