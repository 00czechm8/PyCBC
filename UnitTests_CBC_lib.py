import unittest
import numpy as np
from CBC_lib import (
    get_traj, get_traj_derivative, control_input,
    get_four_coeffs, get_amplitude, estimate_wavelength,
    compute_phase_difference, estimate_dominant_freq
)

class TestCBCLib(unittest.TestCase):

    def setUp(self):
        self.fs = 1000  # Hz
        self.omega = 2 * np.pi * 5  # 5 Hz
        self.t = np.linspace(0, 1, self.fs, endpoint=False)
        self.signal = np.sin(self.omega * self.t)

    def test_get_traj(self):
        coeffs = np.zeros((3, 2))
        coeffs[1, 0] = 1  # cos(omega t)
        traj = get_traj(coeffs, self.omega)
        result = traj(self.t)
        np.testing.assert_allclose(result, np.cos(self.omega * self.t), atol=1e-2)

    def test_get_traj_derivative(self):
        coeffs = np.zeros((3, 2))
        coeffs[1, 0] = 1  # cos(omega t)
        dtraj = get_traj_derivative(coeffs, self.omega)
        result = dtraj(self.t)
        expected = -self.omega * np.sin(self.omega * self.t)
        np.testing.assert_allclose(result, expected, atol=1e-2)

    def test_control_input(self):
        coeffs = np.zeros((3, 2))
        coeffs[1, 0] = 1
        kp = 0.1
        kd = 0.05
        traj = get_traj(coeffs, self.omega)
        dtraj = get_traj_derivative(coeffs, self.omega)
        t_sample = 0.1
        u = control_input(0, 0, traj, dtraj, kp, kd, t_sample)
        expected = kp * traj(t_sample) + kd * dtraj(t_sample)
        self.assertAlmostEqual(u, expected)

    def test_get_four_coeffs(self):
        m = 3
        coeffs = get_four_coeffs(self.signal, m, self.omega, self.fs)
        self.assertEqual(coeffs.shape, (m + 1, 2))
        self.assertAlmostEqual(coeffs[1, 1], 1.0, places=1)  # B1 ≈ -1 for sin(ωt)

    def test_get_amplitude(self):
        amp = get_amplitude(self.signal)
        self.assertAlmostEqual(amp, 1.0, places=1)

    def test_estimate_wavelength(self):
        wl = estimate_wavelength(self.signal, self.fs)
        expected_period = 1 / 5  # since freq = 5 Hz
        self.assertAlmostEqual(wl['seconds'], expected_period, places=2)

    def test_compute_phase_difference(self):
        shifted_signal = np.sin(self.omega * self.t + np.pi / 4)
        phase_diff = compute_phase_difference(self.signal, shifted_signal, self.fs, 1/5)
        self.assertAlmostEqual(phase_diff, -np.pi / 4, delta=0.2)

    def test_estimate_dominant_freq(self):
        freq = estimate_dominant_freq(self.signal, self.fs)
        self.assertAlmostEqual(freq, 5.0, places=0)

if __name__ == '__main__':
    unittest.main()
