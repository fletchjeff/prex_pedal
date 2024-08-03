class Helmholtz {
    constructor(frameSize, overlap, bias = 0.3) {
        this.frameSize = frameSize;
        this.overlap = overlap;
        this.biasFactor = bias;
        this.inputBuf = new Array(frameSize).fill(0);
        this.processBuf = new Array(frameSize * 2).fill(0);
        this.timeIndex = 0;
        this.periodIndex = 0;
        this.periodLength = 0.0;
        this.fidelity = 0.0;
        this.minRms = 0.01;
    }

    iosamples(inputData) {
        const mask = this.frameSize - 1;
        if (!(this.timeIndex & (this.frameSize / this.overlap - 1))) {
            this.analyzeFrame();
        }
        for (let i = 0; i < inputData.length; i++) {
            this.inputBuf[this.timeIndex] = inputData[i];
            this.timeIndex = (this.timeIndex + 1) & mask;
        }
    }

    setFrameSize(frame) {
        this.frameSize = frame;
        this.inputBuf = new Array(frame).fill(0);
        this.processBuf = new Array(frame * 2).fill(0);
        this.timeIndex = 0;
    }

    setOverlap(lap) {
        this.overlap = lap;
    }

    setBias(bias) {
        this.biasFactor = Math.max(0.0, Math.min(1.0, bias));
    }

    setMinRms(rms) {
        this.minRms = Math.max(0.0, Math.min(1.0, rms));
    }

    getPeriod() {
        return this.periodLength;
    }

    getFidelity() {
        return this.fidelity;
    }

    analyzeFrame() {
        const tindex = this.timeIndex;
        const mask = this.frameSize - 1;
        const norm = 1.0 / Math.sqrt(this.frameSize * 2);
        for (let n = 0; n < this.frameSize; n++) {
            this.processBuf[n] = this.inputBuf[(tindex + n) & mask] * norm;
        }
        for (let n = this.frameSize; n < this.frameSize * 2; n++) {
            this.processBuf[n] = 0.0;
        }
        this.autocorrelation();
        this.normalize();
        this.pickPeak();
        this.calculatePeriodAndFidelity();
    }

    autocorrelation() {
        const fftsize = this.frameSize * 2;
        const realFft = (x) => {
            // Implement a simple real FFT function
            const N = x.length;
            const X = new Array(N).fill(0).map(() => [0, 0]);
            for (let k = 0; k < N; k++) {
                for (let n = 0; n < N; n++) {
                    const phi = -2 * Math.PI * k * n / N;
                    X[k][0] += x[n] * Math.cos(phi);
                    X[k][1] += x[n] * Math.sin(phi);
                }
            }
            return X;
        };

        const realIfft = (X) => {
            // Implement a simple real IFFT function
            const N = X.length;
            const x = new Array(N).fill(0);
            for (let n = 0; n < N; n++) {
                for (let k = 0; k < N; k++) {
                    const phi = 2 * Math.PI * k * n / N;
                    x[n] += X[k][0] * Math.cos(phi) - X[k][1] * Math.sin(phi);
                }
                x[n] /= N;
            }
            return x;
        };

        this.processBuf = realFft(this.processBuf);
        this.processBuf[0] = [this.processBuf[0][0] ** 2, 0];
        this.processBuf[this.frameSize] = [this.processBuf[this.frameSize][0] ** 2, 0];
        for (let n = 1; n < this.frameSize; n++) {
            const re = this.processBuf[n][0] ** 2 - this.processBuf[n][1] ** 2;
            const im = 2 * this.processBuf[n][0] * this.processBuf[n][1];
            this.processBuf[n] = [re, im];
        }
        this.processBuf = realIfft(this.processBuf);
    }

    normalize() {
        let rms = this.minRms / Math.sqrt(1.0 / this.frameSize);
        const minRzero = rms * rms;
        let rzero = this.processBuf[0];
        if (rzero < minRzero) {
            rzero = minRzero;
        }
        let normIntegral = rzero * 2.0;
        this.processBuf[0] = 1.0;
        for (let n = 1; n < this.frameSize; n++) {
            normIntegral -= (this.inputBuf[n - 1] ** 2 + this.inputBuf[this.frameSize - n] ** 2);
            this.processBuf[n] /= normIntegral * 0.5;
        }
    }

    pickPeak() {
        let maxValue = 0.0;
        let peakIndex = 0;
        const bias = this.biasFactor / this.frameSize;
        for (let n = 1; n < this.frameSize; n++) {
            if (this.processBuf[n] < 0) {
                break;
            }
            if (this.processBuf[n] > this.processBuf[n - 1] && this.processBuf[n] > this.processBuf[n + 1]) {
                const realPeak = this.interpolate3max(this.processBuf, n);
                if (realPeak * (1.0 - n * bias) > maxValue) {
                    maxValue = realPeak;
                    peakIndex = n;
                }
            }
        }
        this.periodIndex = peakIndex;
    }

    calculatePeriodAndFidelity() {
        if (this.periodIndex) {
            this.periodLength = this.periodIndex + this.interpolate3phase(this.processBuf, this.periodIndex);
            this.fidelity = this.interpolate3max(this.processBuf, this.periodIndex);
        }
    }

    interpolate3max(buf, peakIndex) {
        const a = buf[peakIndex - 1];
        const b = buf[peakIndex];
        const c = buf[peakIndex + 1];
        return b + 0.5 * ((c - a) ** 2) / (2 * b - a - c);
    }

    interpolate3phase(buf, peakIndex) {
        const a = buf[peakIndex - 1];
        const b = buf[peakIndex];
        const c = buf[peakIndex + 1];
        return 0.5 * (c - a) / (2.0 * b - a - c);
    }
}

class BitstreamAutocorrelation {
    constructor(frameSize) {
        this.frameSize = frameSize;
    }

    process(inputData) {
        const bitstream = inputData.map(val => (val >= 0 ? 1 : -1));
        const autocorr = new Array(this.frameSize).fill(0);
        for (let lag = 0; lag < this.frameSize; lag++) {
            for (let i = 0; i < this.frameSize - lag; i++) {
                autocorr[lag] += bitstream[i] * bitstream[i + lag];
            }
        }
        return autocorr;
    }
}

class DualPitchDetector {
    constructor(frameSize, overlap, bias = 0.3) {
        this.helmholtz = new Helmholtz(frameSize, overlap, bias);
        this.bitstream = new BitstreamAutocorrelation(frameSize);
    }

    process(inputData) {
        this.helmholtz.iosamples(inputData);
        const helmholtzPeriod = this.helmholtz.getPeriod();
        const helmholtzFidelity = this.helmholtz.getFidelity();

        const bitstreamAutocorr = this.bitstream.process(inputData);
        const bitstreamPeriod = bitstreamAutocorr.slice(1).indexOf(Math.max(...bitstreamAutocorr.slice(1))) + 1;
        const bitstreamFidelity = bitstreamAutocorr[bitstreamPeriod] / bitstreamAutocorr[0];

        if (helmholtzFidelity > bitstreamFidelity) {
            return { period: helmholtzPeriod, fidelity: helmholtzFidelity };
        } else {
            return { period: bitstreamPeriod, fidelity: bitstreamFidelity };
        }
    }
}

// Example usage
const frameSize = 1024;
const overlap = 2;
const bias = 0.3;

const detector = new DualPitchDetector(frameSize, overlap, bias);
const inputData = Array.from({ length: frameSize }, () => Math.random()); // Example input data
const { period, fidelity } = detector.process(inputData);
console.log(`Period: ${period}, Fidelity: ${fidelity}`);
