import unittest
import numpy as np
from CBC_lib import Backbone

class TestCBCLib(unittest.TestCase):

    def setUp(self):
        self.fs = 1000  # Hz
        self.omega = 2 * np.pi * 5  # 5 Hz
        self.t = np.linspace(0, 1, self.fs, endpoint=False)
        self.signal = np.sin(self.omega * self.t)
        self.cbc = Backbone()
        self.cbc.fs = self.fs

    def test_get_traj(self):
        coeffs = np.zeros((3, 2))
        coeffs[1, 0] = 1  # cos(omega t)
        traj = self.cbc.get_traj(coeffs, self.omega)
        result = traj(self.t)
        np.testing.assert_allclose(result, np.cos(self.omega * self.t), atol=1e-2)

    def test_get_traj_derivative(self):
        coeffs = np.zeros((3, 2))
        coeffs[1, 0] = 1  # cos(omega t)
        dtraj = self.cbc.get_traj_derivative(coeffs, self.omega)
        result = dtraj(self.t)
        expected = -self.omega * np.sin(self.omega * self.t)
        np.testing.assert_allclose(result, expected, atol=1e-2)

    def test_get_four_coeffs(self):
        m = 3
        coeffs = self.cbc.get_four_coeffs(self.signal, m, self.omega, self.fs)
        self.assertEqual(coeffs.shape, (m + 1, 2))
        # For sin(ωt), B1 ≈ -1, A1 ≈ 0
        self.assertAlmostEqual(coeffs[1, 1], -1.0, places=1)
        self.assertAlmostEqual(coeffs[1, 0], 0.0, places=1)

    def test_get_amplitude(self):
        amp = self.cbc.get_amplitude(self.signal)
        self.assertAlmostEqual(amp, 1.0, places=1)

    def test_estimate_wavelength(self):
        wl = self.cbc.estimate_wavelength(self.signal, self.fs)
        expected_period = 1 / 5  # since freq = 5 Hz
        self.assertAlmostEqual(wl['seconds'], expected_period, places=2)

    def test_compute_phase_difference(self):
        shifted_signal = np.sin(self.omega * self.t + np.pi / 4)
        phase_diff = self.cbc.compute_phase_difference(self.signal, shifted_signal)
        # Allow for wrap-around
        self.assertTrue(np.isclose(abs(phase_diff), np.pi / 4, atol=0.2))

    def test_estimate_dominant_freq(self):
        freq = self.cbc.estimate_dominant_freq(self.signal)
        self.assertAlmostEqual(freq, 5.0, places=0)

    def test_segment_signal(self):
        # Use a longer signal to ensure enough samples
        long_signal = np.tile(self.signal, 40)  # 40 seconds
        long_forcing = np.tile(np.cos(self.omega * self.t), 40)
        seg_signal, seg_forcing, wl = self.cbc.segment_signal(long_signal, long_forcing, self.fs)
        self.assertEqual(len(seg_signal), len(seg_forcing))
        self.assertGreater(len(seg_signal), 0)
        self.assertIn('samples', wl)
        self.assertIn('seconds', wl)

    def test_get_backbone_point(self):
        amp, freq = self.cbc.get_backbone_point(self.signal)
        self.assertAlmostEqual(amp, 1.0, places=1)
        self.assertAlmostEqual(freq, 5.0, places=0)

if __name__ == '__main__':
    unittest.main()
