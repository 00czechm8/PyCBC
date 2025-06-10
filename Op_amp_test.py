
import time
import threading
from daqhats import mcc128, AnalogInputRange, AnalogInputMode, mcc152
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# User parameters
mcc152_address = 1
mcc128_address = 0
channel_dac = 0
channel_adc = 0

amplitude = 1.0         # Amplitude of sine wave (V)
frequency = 1.0         # Frequency in Hz
dc_offset = 3         # DC offset (V)
sample_rate = 100.0     # Hz
sample_interval = 1.0 / sample_rate
duration = 10           # Duration in seconds

# Initialize devices
dac = mcc152(mcc152_address)
adc = mcc128(mcc128_address)
adc.a_in_range_write(AnalogInputRange.BIP_10V)
adc.a_in_mode_write(AnalogInputMode.SE)
# Configure DIO0 as output and set it high (5V with external pull-up)
# dac.dio_reset()  # Optional: ensure clean state
# dac.dio_direction_write_bit(0, 1)  # Set DIO0 as output
# dac.dio_write_bit(0, 1)            # Set DIO0 high (logic 1)

dac.a_out_write(channel_dac, 0.0)
dac.a_out_write(1, 2.5)
stop_threads = False
plot_window = 5.0       # seconds of data shown in plot
max_samples = int(plot_window * sample_rate)

# Shared data for plotting
time_data = deque(maxlen=max_samples)
voltage_data = deque(maxlen=max_samples)

def signal_generator():
    print("Starting sine wave output...")
    t0 = time.time()
    while not stop_threads and (time.time() - t0 < duration):
        t = time.time() - t0
        value = dc_offset + amplitude * np.sin(2 * np.pi * frequency * t)
        dac.a_out_write(channel_dac, value)
        time.sleep(sample_interval)
    dac.a_out_write(channel_dac, 0.0)

def adc_reader():
    print("Starting ADC sampling...")
    t0 = time.time()
    while not stop_threads and (time.time() - t0 < duration):
        t = time.time() - t0
        voltage = adc.a_in_read(channel_adc)
        print(voltage,"\n")
        time_data.append(t)
        voltage_data.append(voltage)
        time.sleep(sample_interval)

def live_plot():
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_ylim(-6, 6)
    ax.set_xlim(0, plot_window)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ADC Voltage (V)")
    ax.grid(True)

    while not stop_threads:
        if time_data:
            t0 = time_data[0]
            t_vals = [t - t0 for t in time_data]
            line.set_data(t_vals, voltage_data)
            ax.set_xlim(0, max(t_vals) if t_vals else plot_window)
            ax.figure.canvas.draw()
            ax.figure.canvas.flush_events()
        time.sleep(0.01)

# Threads
adc_thread = threading.Thread(target=adc_reader)
dac_thread = threading.Thread(target=signal_generator)
plot_thread = threading.Thread(target=live_plot)

# Run
adc_thread.start()
dac_thread.start()
plot_thread.start()

dac_thread.join()
adc_thread.join()
stop_threads = True
plot_thread.join()
print("Done.")
plt.ioff()

