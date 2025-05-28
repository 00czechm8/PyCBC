import numpy as np
from scipy.fft import fft
from scipy.signal import find_peaks

# ---------------------------
# Controller Logic
# ---------------------------

def get_traj(Four_coeffs, omega):
    wavenums = np.arange(Four_coeffs.shape[0])
    def x_star(t):
        cos_terms = np.cos(np.outer(t, wavenums * omega))
        sin_terms = np.sin(np.outer(t, wavenums * omega))
        mat = np.hstack([cos_terms, sin_terms])
        coeffs = np.hstack([Four_coeffs[:, 0], Four_coeffs[:, 1]])
        return mat @ coeffs
    return x_star

def get_traj_derivative(Four_coeffs, omega):
    wavenums = np.arange(Four_coeffs.shape[0])
    def x_star_dot(t):
        dcos = -np.sin(np.outer(t, wavenums * omega)) * (wavenums * omega)
        dsin =  np.cos(np.outer(t, wavenums * omega)) * (wavenums * omega)
        mat = np.hstack([dcos, dsin])
        coeffs = np.hstack([Four_coeffs[:, 0], Four_coeffs[:, 1]])
        return mat @ coeffs
    return x_star_dot

def control_input(x, v, x_star_func, x_dot_star_func, kp, kd, t):
    return kp * (x_star_func(t) - x) + kd * (x_dot_star_func(t) - v)

# ---------------------------
# Fourier Tools
# ---------------------------

def get_four_coeffs(signal, m, omega, fs):
    n = len(signal)
    if n % 2 != 0:
        signal = signal[:-1]
        n -= 1
    y = fft(signal)
    y = 2 * y[:n // 2] / n
    f = np.arange(n // 2) * (fs / n) * 2 * np.pi
    harm_f = omega * np.arange(1, m + 1)
    y_interp = np.interp(harm_f, f, y)
    A0 = np.real(y[0])
    A_n = np.real(y_interp)
    B_n = -np.imag(y_interp)
    coeffs = np.zeros((m + 1, 2))
    coeffs[0, 0] = A0
    coeffs[1:, 0] = A_n
    coeffs[1:, 1] = B_n
    return coeffs

def get_amplitude(signal):
    return 0.5 * (np.max(signal) - np.min(signal))

def estimate_wavelength(signal, fs):
    peaks, _ = find_peaks(signal)
    if len(peaks) < 2:
        raise ValueError("Not enough peaks to estimate wavelength.")
    diffs = np.diff(peaks)
    avg_samples = np.mean(diffs)
    return {
        "samples": avg_samples,
        "seconds": avg_samples / fs
    }

def compute_phase_difference(sig1, sig2, fs, period_sec):
    p1, _ = find_peaks(sig1)
    p2, _ = find_peaks(sig2)
    if len(p1) < 1 or len(p2) < 1:
        raise ValueError("Not enough peaks for phase diff")
    delta_samples = p2[0] - p1[0]
    delta_time = delta_samples / fs
    phase_diff_rad = (2 * np.pi * delta_time / period_sec+np.pi) % (2 * np.pi)-np.pi
    return phase_diff_rad

def estimate_dominant_freq(signal, fs):
    y = np.abs(fft(signal))
    peak_idx = np.argmax(y[:len(y)//2])
    return peak_idx * fs / len(signal)

# ---------------------------
# CBC Backbone Tracing Loop
# ---------------------------

# def run_backbone_trace(F_start, h, m, fs, omega_init, delta, alpha, beta, kp, kd, q_tol, e_tol, n_max):
#     Ts = 1 / fs
#     tspan = np.arange(0, 50, Ts)

#     freq_list = []
#     amp_list = []
#     omega = [omega_init, omega_init, omega_init * 3]
#     X_n = np.zeros((m + 1, 2, n_max))
#     X_star = np.zeros((m + 1, 2))

#     n_counter = 0

#     while n_counter < n_max:
#         F = F_start + n_counter * h
#         q_omega = [np.inf, np.inf, np.inf]
#         val_dif = np.inf
#         e_u = np.inf
#         e_counter = 1
#         q_counter = 1

#         while abs(val_dif) > q_tol and q_counter <= 150:
#             while e_u > e_tol and e_counter <= 100:
#                 if e_counter == 1:
#                     X_star[1, 0] = 0
#                     X_star[1, 1] = F

#                 x_star_func = get_traj(X_star, omega[1])
#                 x_dot_star_func = get_traj_derivative(X_star, omega[1])
                
#                 # Placeholder: You must implement this
#                 t_res, y_res = simulate_system(tspan, x_star_func, x_dot_star_func, delta, alpha, beta, omega[1], F, kp, kd)

#                 signal = y_res[:, 0]  # assume first column is x(t)
#                 wl = estimate_wavelength(signal[int(10*fs):], fs)
#                 wl_idx = int(round(wl['samples']))
                
#                 end_idx = np.where(np.abs(signal) <= 1e-3)[0][-1]
#                 num_periods = int(35 / (wl_idx * Ts))
#                 seg_start = end_idx - wl_idx * num_periods
#                 seg = signal[seg_start:end_idx]

#                 Four_coeffs = get_four_coeffs(seg, m, omega[1], fs)
#                 e_u = np.linalg.norm(Four_coeffs[2:] - X_star[2:])
#                 if e_u > e_tol:
#                     X_star[2:] = Four_coeffs[2:]
#                 e_counter += 1

#             forcing = F * np.cos(omega[1] * tspan[seg_start:end_idx])
#             q_omega[2] = compute_phase_difference(forcing, seg, fs, wl['seconds']) - np.pi / 2

#             if q_counter > 2:
#                 val_dif = abs(q_omega[0] - q_omega[2])

#             if abs(val_dif) <= q_tol:
#                 A = get_amplitude(seg)
#                 freq = estimate_dominant_freq(seg, fs)
#                 freq_list.append(freq)
#                 amp_list.append(A)
#                 X_n[:, :, n_counter] = Four_coeffs
#                 if n_counter > 0:
#                     X_star = X_n[:, :, n_counter] + h * (X_n[:, :, n_counter] - X_n[:, :, n_counter - 1])
#                 else:
#                     X_star = X_n[:, :, n_counter]
#                 omega = [2 * np.pi * freq] * 3
#                 break

#             else:
#                 q_omega = [q_omega[1], q_omega[2], q_omega[1] if q_counter == 1 else q_omega[2]]
#                 if q_omega[0] * q_omega[1] < 0:
#                     omega[2] = omega[1]
#                     omega[1] = (omega[0] + omega[2]) / 2
#                 else:
#                     omega[0] = omega[1]
#                     omega[1] = (omega[0] + omega[2]) / 2
#                 e_u = np.inf
#                 e_counter = 1

#             q_counter += 1
#         n_counter += 1

#     return freq_list, amp_list, X_n