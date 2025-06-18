import CBC_lib
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import time

# Instantiate the backbone class
LC = CBC_lib.Backbone()

# Start the DAC and ADC
LC.start_hats()

# Setup internal attributes
# LC.pause_event = multiprocessing.Event()
# LC.stop_event = multiprocessing.Event()
# LC.F_spin_up = multiprocessing.Value("d", 0.05)
# LC.omega_spin_up = multiprocessing.Value("d", 100)
LC.fs = 1e3
F_test = 0.05
omega_test = 10

# LC.pause_event.set()
# spin_up_process = multiprocessing.Process(target=LC.spin_up, args=(LC.F_spin_up.value, LC.omega_spin_up.value, LC.pause_event, LC.stop_event, LC.dac, LC.output_channel))
# spin_up_process.start()

num_samples = 5
time_length = int(num_samples*LC.fs)
sampled_load = np.zeros(num_samples)
forcing_signal = np.zeros(num_samples)

load = np.zeros(time_length)
forcing = np.zeros(time_length)
time_vec = np.zeros(time_length)

start_time = time.time()
target_time = start_time
for idx in range(time_length):

    output = F_test * np.cos(2*np.pi*omega_test * (target_time-start_time))+2.5
    LC.dac.a_out_write(0, output)
    time_vec[idx] = target_time
    print("Time:", idx/LC.fs, "Output:", output)

    for i in range(num_samples):
        sampled_load[i] = (1/11.21)*1e3*LC.adc.a_in_read(LC.load_cell_channel)
        forcing_signal[i] = LC.adc.a_in_read(LC.read_channel)
    load[idx] = np.mean(sampled_load)
    forcing[idx] = np.mean(forcing_signal)
    target_time = start_time + (idx + 1) / LC.fs
    time.sleep(max(0, target_time - time.time()))

# Load Calibration
forcing_amp = LC.get_amplitude(forcing)
load_amp = LC.get_amplitude(load)

forcingV2load_amp = load_amp/forcing_amp
print("Load Amp.:", load_amp, "Forcing Amp.:", forcing_amp)
print("Load Constant:", forcingV2load_amp, ", Pre-load:", np.mean(load))

# Latency
phase_diff = LC.compute_phase_difference(forcing, load)
print("Latency:", (phase_diff/(2*np.pi))/LC.fs)

# LC.stop_event.set()
# if spin_up_process.is_alive():
#     spin_up_process.terminate()

plt.plot(time_vec, load)
plt.plot(time_vec, forcing)
plt.show()


