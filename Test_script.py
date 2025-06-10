import CBC_lib as cbc
import numpy as np

omega_init = 0
F_start = 0
h = 0
m = 0
fs = 0
kp = 0.02
kd = 0.05
q_tol = 0
e_tol = 0
n_max = 0
Ts = 1 / fs
tspan = np.arange(0, 50, Ts)

freq_list = []
amp_list = []
omega = [omega_init, omega_init, omega_init * 3]
X_n = np.zeros((m + 1, 2, n_max))
X_star = np.zeros((m + 1, 2))

n_counter = 0

while n_counter < n_max:
    F = F_start + n_counter * h
    q_omega = [np.inf, np.inf, np.inf]
    val_dif = np.inf
    e_u = np.inf
    e_counter = 1
    q_counter = 1

    while abs(val_dif) > q_tol and q_counter <= 150:
        while e_u > e_tol and e_counter <= 100:
            if e_counter == 1:
                X_star[1, 0] = 0
                X_star[1, 1] = F

            x_star_func = cbc.get_traj(X_star, omega[1])
            x_dot_star_func = cbc.get_traj_derivative(X_star, omega[1])
            
            # RUN SYSTEM HERE
            

            signal = y_res[:, 0]  # assume first column is x(t)
            wl = cbc.estimate_wavelength(signal[int(10*fs):], fs)
            wl_idx = int(round(wl['samples']))
            
            end_idx = np.where(np.abs(signal) <= 1e-3)[0][-1]
            num_periods = int(35 / (wl_idx * Ts))
            seg_start = end_idx - wl_idx * num_periods
            seg = signal[seg_start:end_idx]

            Four_coeffs = cbc.get_four_coeffs(seg, m, omega[1], fs)
            e_u = np.linalg.norm(Four_coeffs[2:] - X_star[2:])
            if e_u > e_tol:
                X_star[2:] = Four_coeffs[2:]
            e_counter += 1

        forcing = F * np.cos(omega[1] * tspan[seg_start:end_idx])
        q_omega[2] = cbc.compute_phase_difference(forcing, seg, fs, wl['seconds']) - np.pi / 2

        if q_counter > 2:
            val_dif = abs(q_omega[0] - q_omega[2])

        if abs(val_dif) <= q_tol:
            A = cbc.get_amplitude(seg)
            freq = cbc.estimate_dominant_freq(seg, fs)
            freq_list.append(freq)
            amp_list.append(A)
            X_n[:, :, n_counter] = Four_coeffs
            if n_counter > 0:
                X_star = X_n[:, :, n_counter] + h * (X_n[:, :, n_counter] - X_n[:, :, n_counter - 1])
            else:
                X_star = X_n[:, :, n_counter]
            omega = [2 * np.pi * freq] * 3
            break

        else:
            q_omega = [q_omega[1], q_omega[2], q_omega[1] if q_counter == 1 else q_omega[2]]
            if q_omega[0] * q_omega[1] < 0:
                omega[2] = omega[1]
                omega[1] = (omega[0] + omega[2]) / 2
            else:
                omega[0] = omega[1]
                omega[1] = (omega[0] + omega[2]) / 2
            e_u = np.inf
            e_counter = 1

        q_counter += 1
    n_counter += 1
