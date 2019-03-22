import math
import pyaudio
import numpy as np
from matplotlib import pyplot as plt
from math import pi

CHUNKSIZE = 4096  # fixed chunk size
RATE = 50000  # Sample rate

# Audio range we want to detect between
minF = 27.5
maxF = 4186
minP = int(RATE / (maxF - 1))
maxP = int(RATE / (minF + 1))


class Wave(object):
    """docstring for Wave"""

    def __init__(self, samplingRate, numSamples, function):
        super(Wave, self).__init__()
        self.numSamples = numSamples
        self.samplingRate = samplingRate
        self.function = function
        self.samples = []
        self.times = []
        self.sampleFunction()

    def sampleFunction(self):
        for t in np.linspace(0, self.samplingRate * self.numSamples, self.numSamples):
            self.times.append(t)
            self.samples.append(self.function(t))


# Test Audio data using my Wave class to generate random samples of a given sine function
def generateSineWave(frequency):
    myWave = Wave(1 / RATE, CHUNKSIZE, lambda x: math.sin(2 * math.pi * frequency * x))
    return np.array(myWave.samples)


def generateMiddleC():
    f = 440 * (2 ** (-0.75))
    p = RATE / f
    n = 2 * maxP
    signal = [0 for x in range(n)]
    for i in range(0, n):
        signal[i] += 1.0 * math.sin(2 * pi * 1 * i / p)
        signal[i] += 0.6 * math.sin(2 * pi * 2 * i / p)
        signal[i] += 0.3 * math.sin(2 * pi * 3 * i / p)
    return signal


def generateComplexWave(f):
    p = RATE / f
    n = 2 * maxP
    signal = [0 for x in range(n)]
    for i in range(0, n):
        signal[i] += 1.0 * math.sin(2 * pi * 1 * i / p)
        signal[i] += 0.6 * math.sin(2 * pi * 2 * i / p)
        signal[i] += 0.3 * math.sin(2 * pi * 3 * i / p)
    return signal


def normalizedAC(signal):
    nac = [0 for x in range(maxP + 2)]
    for p in range(minP - 1, maxP + 2):
        ac, sqSumStart, sqSumEnd = 0, 0, 0
        for i in range(0, len(signal) - p):
            ac += signal[i] * signal[i + p]
            sqSumStart += signal[i] * signal[i]
            sqSumEnd += signal[i + p] * signal[i + p]
        nac[p] = ac / math.sqrt(sqSumStart * sqSumEnd)
    return nac


def fastNAC(signal):
    x = np.asarray(signal)
    N = len(x)
    x = x - x.mean()
    s = np.fft.fft(x, N * 2 - 1)
    result = np.real(np.fft.ifft(s * np.conjugate(s), N * 2 - 1))
    result = result[:N]
    result /= result[0]
    return result


def getPeak(nac):
    peak = minP
    for p in range(minP, maxP + 1):
        if nac[p] > nac[peak]:
            peak = p
    return peak


def correctOctaveErrors(nac, bestP, pEst):
    kThreshold = 0.9
    maxMultiple = int(bestP / minP)
    found = False
    mul = maxMultiple
    while not found and mul >= 1:
        allStrong = True

        for k in range(1, mul):
            subMulPeriod = int(k * pEst / mul + 0.5)
            if nac[subMulPeriod] < kThreshold * nac[bestP]:
                allStrong = False

        if allStrong:
            found = True
            pEst = pEst / mul

        mul -= 1

    return pEst


def estimatePeriod(signal):
    # Tracks the quality of the periodicity of the signal
    q = 0

    # Get the normalized autocorrelation of the signal
    nac = fastNAC(signal)

    # Get the highest peak of the NAC in the range of interest
    bestP = getPeak(nac)

    # If bestP is the highest value, but not the peak, we can't determine the period
    if nac[bestP] < nac[bestP - 1] and nac[bestP] < nac[bestP + 1]:
        return 0

    # Quality of the signal is the NAC at the highest peak
    q = nac[bestP]

    # Interpolate the right and left values to guess the real peak
    left, mid, right = nac[bestP - 1], nac[bestP], nac[bestP + 1]
    shift = mid
    if (2 * mid - left - right) > 0:
        shift = 0.5 * (right - left) / (2 * mid - left - right)

    # Add the shift to the peak value to get the estimated period
    pEst = bestP + shift

    # Account for octave errors by looking through all integer submultiple periods
    pEst = correctOctaveErrors(nac, bestP, pEst)

    # Return the period and quality
    return pEst, q


def detectFundamentalFrequency(signal):
    # Estimate the period
    periodEstimate, quality = estimatePeriod(signal)
    frequencyEstimate = 0
    if periodEstimate > 0:
        frequencyEstimate = RATE / periodEstimate

    return frequencyEstimate


def initializeMicrophone():
    # initialize portaudio
    p = pyaudio.PyAudio()
    return p


def getMicrophoneData(p):
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)

    # do this as long as you want fresh samples
    data = stream.read(CHUNKSIZE)
    numpydata = np.fromstring(data, dtype=np.int16).tolist()

    # close stream
    stream.stop_stream()
    stream.close()

    return numpydata


def closeMicrophone(p):
    p.terminate()


def graphSignal(signal):
    plt.plot(signal)
    plt.figure()
    plt.plot(normalizedAC(signal))
    plt.show(block=False)


def freqToPitch(frequency):
    if 15.5 < frequency < 16.5:
        pitch = "C0"
    elif 16.5 < frequency < 17.5:
        pitch = "C#0"
    elif 17.5 < frequency < 19.0:
        pitch = "D0"
    elif 19.0 < frequency < 20.5:
        pitch = "D#0"
    elif 20.5 < frequency < 21.5:
        pitch = "E0"
    elif 21.5 < frequency < 22.5:
        pitch = "F0"
    elif 22.5 < frequency < 24.0:
        pitch = "F#0"
    elif 24.0 < frequency < 25.5:
        pitch = "G0"
    elif 25.5 < frequency < 27.0:
        pitch = "G#0"
    elif 27.0 < frequency < 28.5:
        pitch = "A0"
    elif 28.5 < frequency < 30.0:
        pitch = "A#0"
    elif 30.0 < frequency < 32.0:
        pitch = "B0"
    elif 32.0 < frequency < 34.0:
        pitch = "C1"
    elif 34.0 < frequency < 36.0:
        pitch = "C#1"
    elif 36.0 < frequency < 38.0:
        pitch = "D1"
    elif 38.0 < frequency < 40.0:
        pitch = "D#1"
    elif 40.0 < frequency < 42.5:
        pitch = "E1"
    elif 42.5 < frequency < 45.0:
        pitch = "F1"
    elif 47.5 < frequency < 50.5:
        pitch = "G1"
    elif 50.5 < frequency < 53.5:
        pitch = "G#1"
    elif 53.5 < frequency < 56.5:
        pitch = "A1"
    elif 56.5 < frequency < 60.0:
        pitch = "A#1"
    elif 60.0 < frequency < 63.5:
        pitch = "B1"
    elif 63.5 < frequency < 67.0:
        pitch = "C2"
    elif 61.5 < frequency < 71.0:
        pitch = "C#2"
    elif 71.0 < frequency < 75.5:
        pitch = "D2"
    elif 75.5 < frequency < 80.0:
        pitch = "D#2"
    elif 80.0 < frequency < 84.5:
        pitch = "E2"
    elif 84.5 < frequency < 90.0:
        pitch = "F2"
    elif 90.0 < frequency < 95.5:
        pitch = "F#2"
    elif 95.5 < frequency < 101.0:
        pitch = "G2"
    elif 101.0 < frequency < 107.0:
        pitch = "G#2"
    elif 107.0 < frequency < 113.5:
        pitch = "A2"
    elif 113.5 < frequency < 120.5:
        pitch = "A#2"
    elif 120.5 < frequency < 127.5:
        pitch = "B2"
    elif 127.5 < frequency < 135.0:
        pitch = "C3"
    elif 135.0 < frequency < 143.0:
        pitch = "C#3"
    elif 143.0 < frequency < 151.5:
        pitch = "D3"
    elif 151.5 < frequency < 160.5:
        pitch = "D#3"
    elif 160.5 < frequency < 170.0:
        pitch = "E3"
    elif 170.0 < frequency < 180.0:
        pitch = "F3"
    elif 180.0 < frequency < 190.5:
        pitch = "F#3"
    elif 190.5 < frequency < 202.0:
        pitch = "G3"
    elif 202.0 < frequency < 214.0:
        pitch = "G#3"
    elif 214.0 < frequency < 226.5:
        pitch = "A3"
    elif 226.5 < frequency < 240.0:
        pitch = "A#3"
    elif 240.0 < frequency < 254.5:
        pitch = "B3"
    elif 254.5 < frequency < 270.0:
        pitch = "C4"
    elif 270.0 < frequency < 286.0:
        pitch = "C#4"
    elif 286.0 < frequency < 302.5:
        pitch = "D4"
    elif 302.5 < frequency < 320.5:
        pitch = "D#4"
    elif 320.5 < frequency < 339.5:
        pitch = "E4"
    elif 339.5 < frequency < 359.5:
        pitch = "F4"
    elif 359.5 < frequency < 381.0:
        pitch = "F#4"
    elif 381.0 < frequency < 403.5:
        pitch = "G4"
    elif 403.5 < frequency < 427.5:
        pitch = "G#4"
    elif 427.5 < frequency < 453.0:
        pitch = "A4"
    elif 453.0 < frequency < 480.0:
        pitch = "A#4"
    elif 480.0 < frequency < 508.5:
        pitch = "B4"
    elif 508.5 < frequency < 538.5:
        pitch = "C5"
    elif 538.5 < frequency < 570.5:
        pitch = "C#5"
    elif 570.5 < frequency < 604.5:
        pitch = "D5"
    elif 604.5 < frequency < 640.5:
        pitch = "D#5"
    elif 640.5 < frequency < 679.0:
        pitch = "E5"
    elif 679.0 < frequency < 719.5:
        pitch = "F5"
    elif 719.5 < frequency < 762.0:
        pitch = "F#5"
    elif 762.0 < frequency < 807.5:
        pitch = "G5"
    elif 807.5 < frequency < 855.5:
        pitch = "G#5"
    elif 855.5 < frequency < 906.0:
        pitch = "A5"
    elif 906.0 < frequency < 960.0:
        pitch = "A#5"
    elif 960.0 < frequency < 1017.5:
        pitch = "B5"
    elif 1017.5 < frequency < 1078.0:
        pitch = "C6"
    elif 1078.0 < frequency < 1142.0:
        pitch = "C#6"
    elif 1142.0 < frequency < 1210.0:
        pitch = "D6"
    elif 1210.0 < frequency < 1282.0:
        pitch = "D#6"
    elif 1282.0 < frequency < 1358.0:
        pitch = "E6"
    elif 1358.0 < frequency < 1436.0:
        pitch = "F6"
    elif 1436.0 < frequency < 1521.5:
        pitch = "F#6"
    elif 1521.5 < frequency < 1614.5:
        pitch = "G6"
    elif 1614.5 < frequency < 1710.5:
        pitch = "G#6"
    elif 1710.5 < frequency < 1812.5:
        pitch = "A6"
    elif 1812.5 < frequency < 1920.5:
        pitch = "A#6"
    elif 1920.5 < frequency < 2034.5:
        pitch = "B6"
    elif 2034.5 < frequency < 2155.5:
        pitch = "C7"
    elif 2155.5 < frequency < 2283.5:
        pitch = "C#7"
    elif 2283.5 < frequency < 2419.0:
        pitch = "D7"
    elif 2419.0 < frequency < 2563.0:
        pitch = "D#7"
    elif 2563.0 < frequency < 2715.5:
        pitch = "E7"
    elif 2715.5 < frequency < 2877.0:
        pitch = "F7"
    elif 2877.0 < frequency < 3048.0:
        pitch = "F#7"
    elif 3048.0 < frequency < 3229.0:
        pitch = "G7"
    elif 3229.0 < frequency < 3421.0:
        pitch = "G#7"
    elif 3421.0 < frequency < 3624.5:
        pitch = "A7"
    elif 3624.5 < frequency < 3840.0:
        pitch = "A#7"
    elif 3840.0 < frequency < 4068.5:
        pitch = "B7"
    elif 4068.5 < frequency < 4310.5:
        pitch = "C8"
    elif 4310.5 < frequency < 4567.0:
        pitch = "C#8"
    elif 4567.0 < frequency < 4838.5:
        pitch = "D8"
    elif 4838.5 < frequency < 5126.0:
        pitch = "D#8"
    elif 5126.0 < frequency < 5431.0:
        pitch = "E8"
    elif 5431.0 < frequency < 5754.0:
        pitch = "F8"
    elif 5754.0 < frequency < 6096.0:
        pitch = "F#8"
    elif 6096.0 < frequency < 6458.5:
        pitch = "G8"
    elif 6458.5 < frequency < 6842.5:
        pitch = "G#8"
    elif 6842.5 < frequency < 7249.5:
        pitch = "A8"
    elif 7249.5 < frequency < 7680.5:
        pitch = "A#8"
    return(pitch)

p = initializeMicrophone()
while True:
    signal = getMicrophoneData(p)
    f = detectFundamentalFrequency(signal)
    if f != 0:
        print(f)
        print(freqToPitch(f))


