import numpy as np
from scipy.fft import fft
from scipy.signal import find_peaks
from daqhats import mcc128, AnalogInputRange, AnalogInputMode, mcc152
import os
import time
from collections import deque
from numba import njit

@njit(cache=True)
def numba_get_four_coeffs(signal, m, omega, fs):
    n = len(signal)
    if n % 2 != 0:
        signal = signal[:-1]
        n -= 1
    y = np.fft.fft(signal)
    y = 2 * y[:n // 2] / n
    f = np.arange(n // 2) * (fs / n) * 2 * np.pi
    harm_f = omega * np.arange(1, m + 1)
    # Numba doesn't support np.interp, so use a simple linear search
    y_interp_real = np.zeros(m)
    y_interp_imag = np.zeros(m)
    for i in range(m):
        idx = np.searchsorted(f, harm_f[i])
        if idx == 0 or idx >= len(f):
            y_interp_real[i] = 0.0
            y_interp_imag[i] = 0.0
        else:
            x0, x1 = f[idx-1], f[idx]
            y0r, y1r = np.real(y[idx-1]), np.real(y[idx])
            y0i, y1i = np.imag(y[idx-1]), np.imag(y[idx])
            frac = (harm_f[i] - x0) / (x1 - x0)
            y_interp_real[i] = y0r + frac * (y1r - y0r)
            y_interp_imag[i] = y0i + frac * (y1i - y0i)
    A0 = np.real(y[0])
    A_n = y_interp_real
    B_n = -y_interp_imag
    coeffs = np.zeros((m + 1, 2))
    coeffs[0, 0] = A0
    coeffs[1:, 0] = A_n
    coeffs[1:, 1] = B_n
    return coeffs

@njit(cache=True)
def numba_traj(t, Four_coeffs, omega):
    wavenums = np.arange(Four_coeffs.shape[0])
    coeffs = np.hstack((Four_coeffs[:, 0], Four_coeffs[:, 1]))
    cos_terms = np.cos(np.outer(t, wavenums * omega))
    sin_terms = np.sin(np.outer(t, wavenums * omega))
    mat = np.hstack((cos_terms, sin_terms))
    return mat @ coeffs

@njit(cache=True)
def numba_traj_derivative(t, Four_coeffs, omega):
    wavenums = np.arange(Four_coeffs.shape[0])
    coeffs = np.hstack((Four_coeffs[:, 0], Four_coeffs[:, 1]))
    omega_wavenums = wavenums * omega
    dcos = -np.sin(np.outer(t, omega_wavenums)) * omega_wavenums
    dsin =  np.cos(np.outer(t, omega_wavenums)) * omega_wavenums
    mat = np.hstack((dcos, dsin))
    return mat @ coeffs

class Backbone:

    def __init__(self):
        self.filename = None
        self.load_cell_constant = None
        self.shaker_constant = None
        self.kp = None
        self.kd = None
        self.dac = None
        self.adc = None
        self.pause_event = None
        self.stop_event = None
        self.output_channel = 0
        self.read_channel = 0
        self.load_cell_channel = 1
        self.reference_channel = 1
        self.ref_voltage = 2.5
        self.fs = None
        self.dopp2vel_constant = 1.0

    def start_hats(self):
        self.dac = mcc152(self.output_channel)
        self.adc = mcc128(self.read_channel)
        self.dac.a_out_write(self.reference_channel, self.ref_voltage)

    def spin_up(self, F_spin_up, omega_spin_up, pause_event, stop_event, address=1, dac_channel=0):
        os.sched_setaffinity(0, {1})
        dac = mcc152(address)
        while not stop_event.is_set():
            pause_event.wait()
            dac.a_out_write(dac_channel, F_spin_up.value * np.cos(omega_spin_up.value * time.time()))
            time.sleep(1/self.fs)
    
    def pause_spin_up(self):
        self.pause_event.clear()
    
    def resume_spin_up(self):
        self.pause_event.set()

    def run_system(self, F, omega, x_star, x_dot_star, duration):
        doppV2vel_const = self.dopp2vel_constant
        shaker_constant = self.shaker_constant
        LC_constant = self.load_cell_constant
        kp = self.kp
        kd = self.kd
        fs = self.fs
        channel_adc = self.read_channel
        channel_dac = self.output_channel
        dac = self.dac
        adc = self.adc
        load_cell = self.load_cell_channel

        t0 = time.time()
        maxlen = int(fs * duration)
        response = np.zeros(maxlen)
        F_act = np.zeros(maxlen)
        time_vec = np.zeros(maxlen)
        Ts = 1 / fs

        num_avg = 5
        # Pre-allocate for averaging
        avg_samples_vel = np.zeros(num_avg)
        avg_samples_LC = np.zeros(num_avg)

        # Initial velocity
        for i in range(num_avg):
            avg_samples_vel[i] = adc.a_in_read(channel_adc)
        dopp_voltage = np.mean(avg_samples_vel)
        velocity = doppV2vel_const * dopp_voltage

        idx = 0
        t_past = t = time.time()

        while idx < maxlen:
            now = time.time()
            t_past = t
            t = now
            time_vec[idx] = t

            # Read velocity and load cell value (vectorized)
            for i in range(num_avg):
                avg_samples_vel[i] = adc.a_in_read(channel_adc)
                avg_samples_LC[i] = adc.a_in_read(load_cell)
            dopp_voltage = np.mean(avg_samples_vel)
            force_voltage = np.mean(avg_samples_LC)
            old_velocity = velocity
            velocity = doppV2vel_const * dopp_voltage
            force = LC_constant * force_voltage
            response[idx] = velocity
            F_act[idx] = force

            # Finite Diff. for PD controller
            accel = (velocity - old_velocity) / (t - t_past) if (t - t_past) > 0 else 0.0

            # Send control update
            elapsed = t - t0
            u = shaker_constant * F * np.cos(2 * np.pi * omega * elapsed) + kp * (x_star(elapsed) - velocity) + kd * (x_dot_star(elapsed) - accel)
            dac.a_out_write(channel_dac, u)
            idx += 1
            time.sleep(Ts)

        return response, F_act, time

    def segment_signal(self, signal, forcing, fs):
        """
        Segments the signal and forcing arrays using the same indices.
        Returns the segmented signal, segmented forcing, and estimated wavelength.
        """
        last_n_samples = int(35 * fs)
        if len(signal) < last_n_samples or len(forcing) < last_n_samples:
            raise ValueError("Signal or forcing shorter than 35 seconds of data")

        recent_signal = signal[-last_n_samples:]
        recent_forcing = forcing[-last_n_samples:]
        wl = self.estimate_wavelength(recent_signal, fs)
        wl_idx = int(round(wl['samples']))
        total_samples = len(recent_signal)
        num_full_periods = total_samples // wl_idx
        max_period_samples = num_full_periods * wl_idx
        seg_candidate_signal = recent_signal[-max_period_samples:]
        seg_candidate_forcing = recent_forcing[-max_period_samples:]

        # Find segment indices based on signal
        start_idx = next((i for i in range(len(seg_candidate_signal)) if abs(seg_candidate_signal[i]) < 0.01), 0)
        end_idx = next((i for i in reversed(range(len(seg_candidate_signal))) if abs(seg_candidate_signal[i]) < 0.01), len(seg_candidate_signal))

        if start_idx >= end_idx:
            raise ValueError("Unable to find proper zero crossings in segment")

        seg_signal = seg_candidate_signal[start_idx:end_idx]
        seg_forcing = seg_candidate_forcing[start_idx:end_idx]
        return seg_signal, seg_forcing, wl

    def get_traj(self, Four_coeffs, omega):
        def x_star(t):
            t = np.atleast_1d(t)
            return numba_traj(t, Four_coeffs, omega)
        return x_star

    def get_traj_derivative(self, Four_coeffs, omega):
        def x_star_dot(t):
            t = np.atleast_1d(t)
            return numba_traj_derivative(t, Four_coeffs, omega)
        return x_star_dot

    def get_four_coeffs(self, signal, m, omega, fs):
        return numba_get_four_coeffs(signal, m, omega, fs)

    def get_amplitude(self, signal):
        return 0.5 * (np.max(signal) - np.min(signal))

    def estimate_wavelength(self, signal, fs):
        peaks, _ = find_peaks(signal)
        if len(peaks) < 2:
            raise ValueError("Not enough peaks to estimate wavelength.")
        avg_samples = np.mean(np.diff(peaks))
        return {"samples": avg_samples, "seconds": avg_samples / fs}

    def compute_phase_difference(self, sig1, sig2):
        fs = self.fs
        p1, _ = find_peaks(sig1)
        p2, _ = find_peaks(sig2)
        if len(p1) < 1 or len(p2) < 1:
            raise ValueError("Not enough peaks for phase diff")
        delta_samples = p2[0] - p1[0]
        delta_time = delta_samples / fs
        period_sec = 1 / fs
        phase_diff_rad = (2 * np.pi * delta_time / period_sec + np.pi) % (2 * np.pi) - np.pi
        return phase_diff_rad

    def estimate_dominant_freq(self, signal):
        y = np.abs(fft(signal))
        half = len(y) // 2
        peak_idx = np.argmax(y[:half])
        return peak_idx * self.fs / len(signal)

    def bisection_point(self, q_omega, omega, q_counter):
        if q_counter == 1:
            omega[0] = omega[1]
            omega[1] = omega[0] * 1.1
        elif q_omega[0] * q_omega[1] > 0 and q_omega[1] < 0:
            omega[1] += 1.1
        elif q_omega[0] * q_omega[2] > 0 and q_omega[0] > 0:
            omega[1] += 0.9
        elif q_omega[0] * q_omega[1] < 0:
            omega[2] = omega[1]
            omega[1] = (omega[0] + omega[2]) / 2
        else:
            omega[0] = omega[1]
            omega[1] = (omega[0] + omega[2]) / 2
        return omega

    def get_backbone_point(self, signal):
        A = self.get_amplitude(signal)
        freq = self.estimate_dominant_freq(signal)
        return A, freq

    def save_to_txt(self, amp_list, freq_list):
        if len(amp_list) != len(freq_list):
            raise ValueError("amp_list and freq_list must be the same length.")
        with open(self.filename, 'w') as f:
            f.write("Amplitude\tFrequency\n")
            for amp, freq in zip(amp_list, freq_list):
                f.write(f"{amp:.17g}\t{freq:.17g}\n")
