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
LC.pause_event = multiprocessing.Event()
LC.stop_event = multiprocessing.Event()
LC.F_spin_up = multiprocessing.Value("d", 0.05)
LC.omega_spin_up = multiprocessing.Value("d", 100)
LC.fs = 1e3

LC.pause_event.set()
spin_up_process = multiprocessing.Process(target=LC.spin_up, args=(LC.F_spin_up.value, LC.omega_spin_up.value, LC.pause_event, LC.stop_event, LC.dac, LC.output_channel))
spin_up_process.start()

num_samples = 5
time_length = int(5*LC.fs)
sampled_load = np.zeros(num_samples)
forcing_signal = np.zeros(num_samples)

load = np.zeros(time_length)
forcing = np.zeros(time_length)

start_time = time.time()
for idx in range(time_length):
    print("Time:", idx/LC.fs)
    for i in range(num_samples):
        sampled_load[i] = LC.adc.a_in_read(LC.load_cell_channel)
        forcing_signal[i] = LC.adc.a_in_read(LC.read_channel)
    load[idx] = np.mean(sampled_load)
    forcing[idx] = np.mean(forcing_signal)
    target_time = start_time + (idx + 1) / LC.fs
    time.sleep(max(0, target_time - time.time()))


LC.stop_event.set()
if spin_up_process.is_alive():
    spin_up_process.terminate()

plt.plot(np.linspace(0, len(load)-1), load)
plt.plot(np.linspace(0, len(forcing)-1), forcing)
plt.show()


