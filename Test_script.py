import CBC_lib
import numpy as np
import os
import multiprocessing
from daqhats import mcc152
import time


def main():

    # Start the backbone object
    cbc = CBC_lib.Backbone()

    # Start connection with hats channels
    cbc.start_hats()

    # Create values in class object
    cbc.filename = "Simple_beam_backbone.txt"
    cbc.kp = 0.02
    cbc.kd = 0.05
    cbc.pause_event = multiprocessing.Event()
    cbc.pause_event.set() # Pause process when pause_spin_up.clear() is called but runs when pause_spin_up.set()
    cbc.stop_event = stop_spin_up = multiprocessing.Event() # to Stop process call stop_spin_up.set() (breaks out of the while loop)
    cbc.output_channel
    cbc.read_channel
    cbc.fs
    cbc.F_spin_up
    cbc.omega_spin_up

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

    
    F_spin_up = multiprocessing.Value("d", F_start)
    omega_spin_up = multiprocessing.Value("d", omega_init)
    spin_up_process = multiprocessing.Process(target=cbc.spin_up, args=(F_spin_up.value, omega_spin_up.value, cbc.pause_event, cbc.stop_event, cbc.dac, cbc.output_channel))
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
                cbc.pause_spin_up()
                signal, F_act = cbc.run_system(F, omega[1], x_star_func, x_dot_star_func, duration) 
                cbc.resume_spin_up()
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
    cbc.dac.a_out_write(0, 0)
    cbc.dac.a_out_write(1, 0)
    cbc.save_to_txt(filename, amp_list, freq_list)

if __name__ == '__main__':
    main()
