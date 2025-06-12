import CBC_lib as cbc
import numpy as np
import os
import multiprocessing
from daqhats import mcc152
import time

def spin_up(F_spin_up, omega_spin_up, pause_event, stop_event):
    # Pin Core
    os.sched_setaffinity(0, {1})
    
    # While process is active
    while not stop_event.is_set():
        # if pause event is set then the 
        pause_event.wait()
        spin_up(F_spin_up.value, omega_spin_up)

def main():
    os.sched_setaffinity(0, {0})
    # Txt file name for saved data at the end
    filename = "Simple_beam_backbone.txt"

    # Important Constants
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
    duration = 60 # trial duration

    freq_list = []
    amp_list = []
    omega = [omega_init, omega_init, omega_init * 3]
    X_n = np.zeros((m + 1, 2, n_max))
    X_star = np.zeros((m + 1, 2))

    n_counter = 0

    dac, adc = cbc.start_hats()
    F_spin_up = multiprocessing.Value("d", F_start)
    omega_spin_up = multiprocessing.Value("d", omega_init)
    pause_spin_up = multiprocessing.Event()
    stop_spin_up = multiprocessing.Event() # to Stop process call stop_spin_up.set() (breaks out of the while loop)
    pause_spin_up.set() # Pause process when pause_spin_up.clear() is called but runs when pause_spin_up.set()
    spin_up_process = multiprocessing.Process(target=cbc.spin_up, args=(F_spin_up.value, omega_spin_up.value, pause_event, stop_event, 1))
    spin_up_process.start()

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
                pause_spin_up.clear()
                signal, F_act = cbc.run_system(adc, dac, F, omega[1], x_star_func, x_dot_star_func, kp, kd, duration) 
                pause_spin_up.set()
                time.sleep(2)
                
                seg_signal, wl = cbc.segment_signal(signal, fs)
                Four_coeffs = cbc.get_four_coeffs(seg_signal, m, omega[1], fs)
                e_u = np.linalg.norm(Four_coeffs[2:] - X_star[2:]) + np.linalg.norm(Four_coeffs[0] - X_star[0])
                if e_u > e_tol:
                    X_star[2:] = Four_coeffs[2:]
                    X_star[0] = Four_coeffs[0]
                e_counter += 1

            forcing = F_act * np.cos(omega[1] * tspan[seg_start:end_idx])
            q_omega[1] = cbc.compute_phase_difference(forcing, seg_signal, fs, wl['seconds'])

            if abs(q_omega[1]) <= q_tol:
                A, freq = cbc.get_backbone_point(seg_signal, fs)
                amp_list.append(A)
                freq_list.append(freq)

            omega = cbc.bisection_point(q_omega, omega, q_counter)

            q_counter += 1
        n_counter += 1
    dac.a_out_write(0, 0)
    dac.a_out_write(1, 0)
    cbc.save_to_txt(filename, amp_list, freq_list)

if __name__ == '__main__':
    main()
