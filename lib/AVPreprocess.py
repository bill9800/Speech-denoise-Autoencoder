import numpy as np
import librosa

# windowing fft/ifft function
def stft(data, fft_size, step_size):
    # short time fourier transform
    window = np.hamming(fft_size)
    win_num = (len(data) - 2 * fft_size) // step_size
    out = np.ndarray((win_num, fft_size), dtype=data.dtype)
    for i in range(win_num):
        left = int(i * step_size)
        right = int(left + fft_size)
        out[i] = data[left: right] * window
    F = np.fft.rfft(out, axis=1)
    return F

def istft(F, fft_size, step_size):
    # inverse short time fourier transform
    data = np.fft.irfft(F, axis=-1)
    window = np.hamming(fft_size)
    number_windows = F.shape[0]
    T = np.zeros((number_windows * step_size + fft_size))
    for i in range(number_windows):
        head = int(i * step_size)
        tail = int(head + fft_size)
        T[head:tail] = T[head:tail] + data[i, :] * window
    return T

# combine FFT bins to mel frequency bins
def mel2freq(mel_data,sr,fft_size,n_mel,fmax=8000):
    matrix= librosa.filters.mel(sr, fft_size, n_mel, fmax=fmax)
    return np.dot(mel_data,matrix)

def freq2mel(f_data,sr,fft_size,n_mel,fmax=8000):
    pre_matrix = librosa.filters.mel(sr, fft_size, n_mel, fmax=fmax)
    matrix = pre_matrix.T / np.sum(pre_matrix.T,axis=0)
    return np.dot(f_data,matrix)

# directly time to mel domain transformation
def time_to_mel(data,sr,fft_size,n_mel,step_size,fmax=8000):
    F = stft(data,fft_size,step_size)
    M = freq2mel(F,sr,fft_size,n_mel,fmax=8000)
    return M

def mel_to_time(M,sr,fft_size,n_mel,step_size,fmax=8000):
    F = mel2freq(M,sr,fft_size,n_mel)
    T = istft(F,fft_size,step_size)
    return T

def real_imag_expand(mel_data):
    # expand the complex data to 2X data with true real and image number
    D = np.zeros((mel_data.shape[0],mel_data.shape[1]*2))
    D[:,::2] = np.real(mel_data)
    D[:,1::2] = np.imag(mel_data)
    return D

# normalization function
def min_max_norm(x):
    # x should be numpy M*N matrix , normalize the N axis
    return (x-np.min(x,axis=0)) / (np.max(x,axis=0)-np.min(x,axis=0))






















