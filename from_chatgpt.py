import numpy as np
from scipy.fftpack import fft, ifft

class Helmholtz:
    def __init__(self, frame_size, overlap, bias=0.3):
        self.frame_size = frame_size
        self.overlap = overlap
        self.bias_factor = bias
        self.input_buf = np.zeros(frame_size)
        self.process_buf = np.zeros(frame_size * 2)
        self.time_index = 0
        self.period_index = 0
        self.period_length = 0.0
        self.fidelity = 0.0
        self.min_rms = 0.01

    def iosamples(self, input_data):
        mask = self.frame_size - 1
        if not (self.time_index & (self.frame_size // self.overlap - 1)):
            self.analyze_frame()
        for i in range(len(input_data)):
            self.input_buf[self.time_index] = input_data[i]
            self.time_index = (self.time_index + 1) & mask

    def set_frame_size(self, frame):
        self.frame_size = frame
        self.input_buf = np.zeros(frame)
        self.process_buf = np.zeros(frame * 2)
        self.time_index = 0

    def set_overlap(self, lap):
        self.overlap = lap

    def set_bias(self, bias):
        self.bias_factor = max(0.0, min(1.0, bias))

    def set_min_rms(self, rms):
        self.min_rms = max(0.0, min(1.0, rms))

    def get_period(self):
        return self.period_length

    def get_fidelity(self):
        return self.fidelity

    def analyze_frame(self):
        tindex = self.time_index
        mask = self.frame_size - 1
        norm = 1.0 / np.sqrt(self.frame_size * 2)
        for n in range(self.frame_size):
            self.process_buf[n] = self.input_buf[(tindex + n) & mask] * norm
        for n in range(self.frame_size, self.frame_size * 2):
            self.process_buf[n] = 0.0
        self.autocorrelation()
        self.normalize()
        self.pick_peak()
        self.calculate_period_and_fidelity()

    def autocorrelation(self):
        fftsize = self.frame_size * 2
        self.process_buf = fft(self.process_buf)
        self.process_buf[0] = self.process_buf[0] ** 2
        self.process_buf[self.frame_size] = self.process_buf[self.frame_size] ** 2
        for n in range(1, self.frame_size):
            self.process_buf[n] = (self.process_buf[n] * np.conj(self.process_buf[n]) +
                                   self.process_buf[fftsize - n] * np.conj(self.process_buf[fftsize - n])).real
            self.process_buf[fftsize - n] = 0.0
        self.process_buf = ifft(self.process_buf).real

    def normalize(self):
        rms = self.min_rms / np.sqrt(1.0 / self.frame_size)
        min_rzero = rms * rms
        rzero = self.process_buf[0]
        if rzero < min_rzero:
            rzero = min_rzero
        norm_integral = rzero * 2.0
        self.process_buf[0] = 1.0
        for n in range(1, self.frame_size):
            norm_integral -= (self.input_buf[n - 1] ** 2 + self.input_buf[self.frame_size - n] ** 2)
            self.process_buf[n] /= norm_integral * 0.5

    def pick_peak(self):
        max_value = 0.0
        peak_index = 0
        bias = self.bias_factor / self.frame_size
        for n in range(1, self.frame_size):
            if self.process_buf[n] < 0:
                break
            if self.process_buf[n] > self.process_buf[n - 1] and self.process_buf[n] > self.process_buf[n + 1]:
                real_peak = self.interpolate_3max(self.process_buf, n)
                if real_peak * (1.0 - n * bias) > max_value:
                    max_value = real_peak
                    peak_index = n
        self.period_index = peak_index

    def calculate_period_and_fidelity(self):
        if self.period_index:
            self.period_length = self.period_index + self.interpolate_3phase(self.process_buf, self.period_index)
            self.fidelity = self.interpolate_3max(self.process_buf, self.period_index)

    @staticmethod
    def interpolate_3max(buf, peak_index):
        a = buf[peak_index - 1]
        b = buf[peak_index]
        c = buf[peak_index + 1]
        return b + 0.5 * ((c - a) ** 2) / (2 * b - a - c)

    @staticmethod
    def interpolate_3phase(buf, peak_index):
        a = buf[peak_index - 1]
        b = buf[peak_index]
        c = buf[peak_index + 1]
        return 0.5 * (c - a) / (2.0 * b - a - c)


class BitstreamAutocorrelation:
    def __init__(self, frame_size):
        self.frame_size = frame_size

    def process(self, input_data):
        bitstream = np.sign(input_data)
        autocorr = np.correlate(bitstream, bitstream, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        return autocorr


class DualPitchDetector:
    def __init__(self, frame_size, overlap, bias=0.3):
        self.helmholtz = Helmholtz(frame_size, overlap, bias)
        self.bitstream = BitstreamAutocorrelation(frame_size)

    def process(self, input_data):
        self.helmholtz.iosamples(input_data)
        helmholtz_period = self.helmholtz.get_period()
        helmholtz_fidelity = self.helmholtz.get_fidelity()

        bitstream_autocorr = self.bitstream.process(input_data)
        bitstream_period = np.argmax(bitstream_autocorr[1:]) + 1
        bitstream_fidelity = bitstream_autocorr[bitstream_period] / bitstream_autocorr[0]

        if helmholtz_fidelity > bitstream_fidelity:
            return helmholtz_period, helmholtz_fidelity
        else:
            return bitstream_period, bitstream_fidelity


# Example usage
frame_size = 1024
overlap = 2
bias = 0.3

detector = DualPitchDetector(frame_size, overlap, bias)
input_data = np.random.rand(frame_size)  # Example input data
period, fidelity = detector.process(input_data)
print(f"Period: {period}, Fidelity: {fidelity}")
