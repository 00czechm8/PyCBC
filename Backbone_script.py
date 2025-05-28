import time
import os
import multiprocessing as mp
import numpy as np
from numba import njit
from daqhats import mcc128, mcc152, AnalogInputMode, OptionFlags

# Config
SAMPLE_RATE = 1000           # Hz
PWM_CHANNEL = 0
ANALOG_CHANNEL = 0
DAC_RANGE = 5.0              # 0–5V output
KP, KI, KD = 2.0, 0.1, 0.01  # Example PID gains

# Utility: Core pinning
def pin_process_to_core(core_id):
    try:
        os.sched_setaffinity(0, {core_id})
    except AttributeError:
        pass  # Not available on all platforms

# Numba-accelerated PID
@njit
def pid_controller(error, integral, derivative, dt, kp, ki, kd):
    return kp * error + ki * integral + kd * derivative

# Worker 1: ADC Sampling + PID + DAC Output (real-time loop)
def control_loop(stop_event):
    pin_process_to_core(1)
    adc = mcc128(0)
    dac = mcc152(0)
    dac.enable_dac(PWM_CHANNEL)
    dac.dac_write(PWM_CHANNEL, 0.0)

    setpoint = 2.5  # volts
    integral = 0.0
    last_error = 0.0
    last_time = time.perf_counter()

    while not stop_event.is_set():
        now = time.perf_counter()
        dt = now - last_time
        if dt < 1.0 / SAMPLE_RATE:
            time.sleep((1.0 / SAMPLE_RATE) - dt)
            continue
        last_time = now

        # Read voltage
        measured = adc.a_in_read(ANALOG_CHANNEL, AnalogInputMode.SE, OptionFlags.DEFAULT)

        # PID error
        error = setpoint - measured
        integral += error * dt
        derivative = (error - last_error) / dt if dt > 0 else 0.0
        last_error = error

        # PID output
        output = pid_controller(error, integral, derivative, dt, KP, KI, KD)

        # Clamp to 0–DAC_RANGE
        output = max(0.0, min(output, DAC_RANGE))

        # Output control signal
        dac.dac_write(PWM_CHANNEL, output)

# Main process management
if __name__ == "__main__":
    stop_event = mp.Event()
    proc = mp.Process(target=control_loop, args=(stop_event,), name="RealTimeControl")

    try:
        proc.start()
        print("Real-time control loop running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        stop_event.set()
    finally:
        proc.join()
        print("Shutdown complete.")
