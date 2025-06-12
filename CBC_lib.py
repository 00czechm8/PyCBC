import numpy as np
from scipy.fft import fft
from scipy.signal import find_peaks
from daqhats import mcc128, AnalogInputRange, AnalogInputMode, mcc152
import os
import time
from collections import deque

class Backbone:

    def __init__(self):
        self.filename
        self.load_cell_constant
        self.shaker_constant
        self.kp
        self.kd
        self.dac
        self.adc
        self.pause_event
        self.stop_event
        self.output_channel
        self.read_channel
        self.fs
    # ---------------------------
    # Controller Logic
    # ---------------------------

    def load_cell_calibration(dac, adc, dac_channel=0, adc_channel=0):
        
        duration = 10
        t0 = time.time()
        t=t0
        while t < duration:
            dac.a_out_write(dac_channel, np.cos())

    def start_hats():
        dac = mcc152(1)
        adc = mcc128(0)
        dac.a_out_write(1, 2.5)
        return dac, adc

    def spin_up(F_spin_up, omega_spin_up, pause_event, stop_event, address=1, dac_channel=0):
        # Pin Core
        os.sched_setaffinity(0, {1})
        dac = mcc152(address)
        # While process is active
        while not stop_event.is_set():
            # if pause event is set then the 
            pause_event.wait()
            dac.a_out_write()
            spin_up(F_spin_up.value, omega_spin_up)
        
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

    def forcing_amp_recieved():
        # Write a method here to measure the load cell forcing
        return None

    def doppV2vel(voltage):
        return voltage

    def doppV2disp(voltage):
        return voltage

    def Newt2V(Force)
        return

    def run_system(dac, adc, F, omega, x_star, x_dot_star, kp, kd, fs, duration, channel_dac=0, channel_adc=0):
        t0 = time.time()
        response = deque(max_len=fs*duration)
        dopp_voltage = adc.a_in_read(0)
        velocity = doppV2vel(dopp_voltage)
        t = t0
        Ts = 1/fs    
        while t < duration:
            t = time.time()-t0
            old_velocity = velocity
            dopp_voltage = adc.a_in_read(channel_adc)
            velocity = doppV2vel(dopp_voltage)
            accel = velocity-old_velocity
            
            dac.a_out_write(channel_dac,  Newt2V(F)*np.cos(2*np.pi*omega*t)+control_input(velocity, accel, x_star, x_dot_star, kp, kd, t))

            time.sleep(Ts)
            
        
        return response, F_act


    # ---------------------------
    # Fourier Tools
    # ---------------------------
    def segment_signal(signal, fs):
        Ts = 1/fs
        wl = estimate_wavelength(signal[int(10*fs):], fs)
        wl_idx = int(round(wl['samples']))
        
        end_idx = np.where(np.abs(signal) <= 1e-3)[0][-1]
        num_periods = int(35 / (wl_idx * Ts))
        seg_start = end_idx - wl_idx * num_periods
        seg = signal[seg_start:end_idx]
        return seg, wl

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
    # FPI Tools
    # ---------------------------

    def bisection_point(q_omega, omega, q_counter):
        # Skip bisection for first iteration to create a bounding region
        if q_counter == 1:
            omega[0] = omega[1]
            omega[1] = omega[0] * 1.1
        
        # In case bounding region guess is wrong, iterate until valid region is found
        elif q_omega[0] * q_omega[1] > 0 and q_omega[1] < 0:
            omega[1] = omega[1]+1.1
        
        elif q_omega[0] * q_omega[2] > 0 and q_omega[0] > 0:
            omega[1] = omega[1]+0.9
        
        # If function has root in left side of region, update bound so middle is the new right bound
        elif q_omega[0] * q_omega[1] < 0:
            omega[2] = omega[1]
            omega[1] = (omega[0] + omega[2]) / 2
        
        # If function has root on right side of region, update bound so middle is the new left bound
        else:
            omega[0] = omega[1]
            omega[1] = (omega[0] + omega[2]) / 2
            
        return omega

    def get_backbone_point(signal, fs):
        A = get_amplitude(signal)
        freq = estimate_dominant_freq(signal, fs)
        return A, freq

    def save_to_txt(filename, amp_list, freq_list):
        if len(amp_list) != len(freq_list):
            raise ValueError("amp_list and freq_list must be the same length.")

        with open(filename, 'w') as f:
            f.write("Amplitude\tFrequency\n")
            for amp, freq in zip(amp_list, freq_list):
                f.write(f"{amp:.17g}\t{freq:.17g}\n")

