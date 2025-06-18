import CBC_lib
import numpy as np
import matplotlib.pyplot as plt
import time

LC = CBC_lib.Backbone()
LC.start_hats()

LC.fs = 10_000  # 10 kHz
F_test = 1
omega_test = 100
dt = 1.0 / LC.fs

# Preallocate
time_length = int(1 * LC.fs)
load = np.zeros(time_length)
forcing = np.zeros(time_length)
time_vec = np.zeros(time_length)

# Rolling average buffer
num_avg = 5
load_buf = np.zeros(num_avg)
force_buf = np.zeros(num_avg)

# Precompute omega_dt to avoid repeated float mult
omega_dt = 2 * np.pi * omega_test * dt

# Incremental cos calculation
cos_val = np.cos(0)
sin_val = np.sin(0)
cos_omega_dt = np.cos(omega_dt)
sin_omega_dt = np.sin(omega_dt)

# Complex rotation method for cosine generation
cos_complex = complex(cos_val, sin_val)
rotator = complex(cos_omega_dt, sin_omega_dt)

start_time = time.perf_counter()
target_time = start_time
print(start_time)
for idx in range(time_length):
    # Cosine update
    cos_complex *= rotator
    signal = F_test * cos_complex.real + 2.5
    LC.dac.a_out_write(0, signal)

    # One ADC read per channel
    load[idx] = (1 / 11.21) * 1e3 * LC.adc.a_in_read(LC.load_cell_channel)
    forcing[idx] = LC.adc.a_in_read(LC.read_channel)
    time_vec[idx] = time.perf_counter() - start_time

    # Precise busy-wait timing
    target_time += dt
    while time.perf_counter() < target_time:
        pass

end_time = time.perf_counter()

print("Total Time:", end_time-start_time)
# ---- Post-processing ---- #
forcing_amp = LC.get_amplitude(forcing)
load_amp = LC.get_amplitude(load)

forcingV2load_amp = load_amp / forcing_amp
print("Load Amp.:", load_amp, "Forcing Amp.:", forcing_amp)
print("Load Constant:", forcingV2load_amp, ", Pre-load:", np.mean(load))
print("Max Load:", np.max(load), "Min. Load:", np.min(load))

# Phase/latency
phase_diff = LC.compute_phase_difference(forcing, load)
print("Latency:", (phase_diff / (2 * np.pi)) / LC.fs)

# Save and plot
plt.plot(time_vec, load)
plt.show()

LC.save_array(load, "Load_0p1.txt")
