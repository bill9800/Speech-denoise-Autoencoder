import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from sklearn.model_selection import train_test_split

# option
DOWNLOAD = 0 # download and preprocess the data
WAVE_PLOT = 0 # plot original wave, noise wave, mixed wave
INVERSE_CHECK = 0 # check the inverse function of mel
DUMP = 0 # dump wave data to real wav
TRAIN_DENOISE = 1# train the denoising model with mel freq input and output
DENOISE = 1 # use the pretrained denoise autoencoder

# command line functions #
def c_mkdir(dir_name):
    # make directory use command line
    command = 'mkdir ' + dir_name
    os.system(command)

def c_download(loc,name,link):
    # use youtube-dl
    command = 'cd %s;' % loc
    command += 'youtube-dl -x --audio-format wav -o '+ name +'.wav ' + link + ';'
    command += 'ffmpeg -i %s.wav -ar 48000 -ac 1 o%s.wav' % (name,name)
    os.system(command)

def c_cut(loc,term,start_time,end_time):
    # cut the audio
    command = 'cd %s;' % loc
    command += 'sox o%s.wav trim%s.wav trim %s %s' % (term,term,start_time,end_time)
    os.system(command)

def c_mix(loc):
    # mix the data in the mix_list
    command = 'cd %s;' % loc
    command += 'sox --combine concatenate trim*.wav -o concat.wav'
    os.system(command)

###################################################################
if DOWNLOAD:
    c_mkdir('clean')
    c_mkdir('noise')

    clean_video_list = ['https://www.youtube.com/watch?v=u4ZoJKF_VuA&t=75s',
                        'https://www.youtube.com/watch?v=gN9dlisaQVM',
                        'https://www.youtube.com/watch?v=c0KYU2j0TM4',
                        'https://www.youtube.com/watch?v=8S0FDjFBj8o']

    for i in range(len(clean_video_list)):
        c_download('clean',str(i),clean_video_list[i])


    start_time = "60"
    end_time = "180"
    for i in range(len(clean_video_list)):
        c_cut('clean',i,start_time,end_time)

    c_mix('clean')

    noise_video_list = ['https://www.youtube.com/watch?v=BOdLmxy06H0']

    for i in range(len(noise_video_list)):
        c_download('noise', str(i), noise_video_list[i])

    noise_s_time = "60"
    noise_e_time = "540"
    c_cut('noise','0',noise_s_time,noise_e_time)

####################################################################

# mix clean audio and noise audio
with open('clean/concat.wav', 'rb') as f:
    clean_data, clean_sr = librosa.load('clean/concat.wav', sr=None)  # time series data,sample rate
with open('noise/trim0.wav', 'rb') as f:
    noise_data, noise_sr = librosa.load('noise/trim0.wav', sr=None)  # time series data,sample rate


# normalize expand the noise
noise_max = np.max(noise_data)
expand_rate = 1/noise_max
noise_data = noise_data*expand_rate

assert clean_sr == noise_sr
mix_data = clean_data*0.7 + noise_data*0.3
mix_sr = clean_sr

####################################################################
if WAVE_PLOT:
    # plot orignial wave
    size = clean_data.shape[0]
    time = np.arange(0,size)*(1.0 / clean_sr)
    plt.figure(1)
    plt.plot(time,clean_data)
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("original wavedata")
    plt.grid('on')

    # plot noise wave
    size = noise_data.shape[0]
    time = np.arange(0,size)*(1.0 / noise_sr)
    plt.figure(2)
    plt.plot(time,noise_data)
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("noise wavedata")
    plt.grid('on')

    # plot mix wave
    # plot orignial wave
    size = mix_data.shape[0]
    time = np.arange(0,size)*(1.0 / mix_sr)
    plt.figure(3)
    plt.plot(time,mix_data)
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("mixed wavedata")
    plt.grid('on')
    plt.show()
######################################################################
if DUMP:
    if(os.path.isfile('mix') == False):
        os.system('mkdir mix')
    wavfile.write("mix/mix.wav",mix_sr,mix_data)

#####################################################################
# convert time data to frequency data

# fft windowing parameter #
fft_size = 1024
step_size = fft_size // 3# distance to slide along the window

# fequency to mel parameter #
n_mels = 80 # number of mel frequency
start_freq = 0.0
end_freq = 8000.0

def window_fft(data,fft_size,step_size):
    window = np.hamming(fft_size)
    number_windows = (data.shape[0]-2*fft_size)//step_size
    output = np.ndarray((number_windows,fft_size),dtype=data.dtype)

    for i in range(number_windows):
        head = int(i*step_size)
        tail = int(head+fft_size)
        output[i] = data[head:tail]*window

    print("shape of output after windowing:",output.shape)

    F = np.fft.rfft(output,axis=-1)
    print("shape of output after windowing and rfft:",F.shape)
    # F = F[:,:F.shape[1]//2]
    # F = np.abs(F)
    return F

def rev_window_fft(F,fft_size,step_size):
    data = np.fft.irfft(F,axis=-1)
    print("shape of IRFFT:",data.shape)
    window = np.hamming(fft_size)
    number_windows = F.shape[0]

    T = np.zeros((number_windows*step_size+fft_size))
    for i in range(number_windows):
        head = int(i*step_size)
        tail = int(head+fft_size)
        T[head:tail] = T[head:tail]+data[i,:]  *window
    return T

# mel2freq_matrix generation
mel2freq_matrix= librosa.filters.mel(mix_sr, fft_size, n_mels, fmax=end_freq)
print("shape of freq2mel:",mel2freq_matrix.shape)
#mel2freq_matrix = np.linalg.pinv(freq2mel_matrix)
freq2mel_matrix= mel2freq_matrix.T / np.sum(mel2freq_matrix.T,axis = 0)
print("shape of mel2freq:",mel2freq_matrix.shape)

# implement mel transfer
fft_mix_data = window_fft(mix_data,fft_size,step_size)
mel_mix_data = np.dot(fft_mix_data,freq2mel_matrix)
fft_clean_data = window_fft(clean_data,fft_size,step_size)
mel_clean_data = np.dot(fft_clean_data,freq2mel_matrix)
print("shape of output after fft_to_mel:",mel_mix_data.shape)
print("shape of output after fft_to_mel:",mel_clean_data.shape)

# implement mel to time (just to check inverse function)
if INVERSE_CHECK:
    test = np.dot(mel_mix_data,mel2freq_matrix)
    ifft_mix_data = rev_window_fft(test,fft_size,step_size)
    plt.figure()
    plt.plot(ifft_mix_data)
    plt.show()
    Tint = ifft_mix_data/max(ifft_mix_data)*32767
    wavfile.write("mix/test.wav",mix_sr,Tint.astype('int16'))


if WAVE_PLOT:
    fig, ax = plt.subplots()
    im = ax.imshow(mel2freq_matrix)
    # Loop over data dimensions and create text annotations.
    for i in range(mel2freq_matrix.shape[0]):
        for j in range(mel2freq_matrix.shape[1]):
            text = ax.text(i, j, "%.1f" % mel2freq_matrix[i, j],
                           ha="center", va="center", color="w")
    fig.tight_layout()
    plt.show()

    plt.plot(fft_mix_data[6000])
    plt.plot(mel_mix_data[6000])
    plt.show()

###################################################################
#split data

# separate real and image number
mel_mix_data = np.dot(fft_mix_data,freq2mel_matrix)
D_X = np.zeros((mel_mix_data.shape[0],mel_mix_data.shape[1]*2))
D_X[:,::2] = np.real(mel_mix_data ) # real number
D_X[:,1::2] = np.imag(mel_mix_data ) # image number
max = np.max(np.abs(D_X))
D_X = D_X / max

mel_clean_data = np.dot(fft_clean_data,freq2mel_matrix)
D_y = np.zeros((mel_clean_data.shape[0],mel_clean_data.shape[1]*2))
D_y[:,::2] = np.real(mel_clean_data)
D_y[:,1::2] = np.imag(mel_clean_data)
D_y = D_y / max

# separate data to train test sets
D_X_train = D_X[:int(D_X.shape[0]*0.9),:]
D_y_train = D_y[:int(D_y.shape[0]*0.9),:]

X_test = D_X[int(D_X.shape[0]*0.9):,:]
y_test = D_y[int(D_y.shape[0]*0.9):,:]

X_train, X_val, y_train, y_val = train_test_split(D_X_train, D_y_train, test_size=0.15, random_state=87)

# Denoise autoencoder model #

## import keras modules
from keras.layers import BatchNormalization,Dropout,Dense,Input,LeakyReLU
from keras import backend as K
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras.models import Model
from keras.utils import plot_model
from keras.initializers import he_normal
from keras.models import model_from_json
from keras import optimizers

if TRAIN_DENOISE:
    n_input_dim = X_train.shape[1]
    n_output_dim = y_train.shape[1]

    n_hidden1 = 500
    n_hidden2 = 1000

    # H1,H3 shares the same weights
    InputLayer1 = Input(shape=(n_input_dim,),name="InputLayer")
    InputLayer2 = BatchNormalization(axis=1,momentum=0.6)(InputLayer1)
    #InputLayer3 = Dropout(0.2)(InputLayer2)

    HiddenLayer1_1 = Dense(n_hidden1,name="H1",activation='relu',kernel_initializer=he_normal(seed=27))(InputLayer2)
    HiddenLayer1_2 = BatchNormalization(axis=1,momentum=0.6)(HiddenLayer1_1)
    HiddenLayer1_3 = Dropout(0.2)(HiddenLayer1_2)

    HiddenLayer2_1 = Dense(n_hidden2,name="H2",activation='relu',kernel_initializer=he_normal(seed=42))(HiddenLayer1_3)

    HiddenLayer3_1 = Dense(n_hidden1,name="H3",activation='relu',kernel_initializer=he_normal(seed=27))(HiddenLayer2_1)
    HiddenLayer3_2 = BatchNormalization(axis=1,momentum=0.6)(HiddenLayer3_1)
    HiddenLayer3_3 = Dropout(0.2)(HiddenLayer3_2)

    OutputLayer= Dense(n_output_dim,name="OutputLayer",kernel_initializer=he_normal(seed=62))(HiddenLayer3_3)

    model = Model(inputs=[InputLayer1],outputs=[OutputLayer])
    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.001, amsgrad=False)
    model.compile(loss='mse',optimizer=opt)

    plot_model(model,to_file='model.png',show_shapes=True,show_layer_names=True)
    model.summary()

    tensorboard = TensorBoard(log_dir="./logs", histogram_freq=0, write_graph=True ,write_images=True)
    # fit the model
    hist = model.fit(X_train, y_train, batch_size= 512, epochs=200, verbose=1, validation_data=([X_val], [y_val]),
                     callbacks=[tensorboard])


    plt.figure(figsize=(10,8))
    plt.plot(hist.history['loss'], label='Loss')
    plt.plot(hist.history['val_loss'],label='Val_Loss')
    plt.legend(loc='best')
    plt.title('Training Loss and Validation Loss')
    plt.show()


    results = model.evaluate(X_test,y_test,batch_size=len(y_test))
    print('Test loss:%3f'%results)


    # serialize model to JSON
    model_json = model.to_json()
    with open("model/model3.json",'w') as f:
        f.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model/model3.h5")
    print("Saved model to disk")

    # use the denoiser
if DENOISE:
    # load josn and create model
    with open('model/model3.json','r') as f:
        loaded_model_json = f.read()
    denoise_model = model_from_json(loaded_model_json)
    denoise_model.load_weights("model/model3.h5")
    print("Loaded model from disk")

    y_pred = denoise_model.predict(D_X)
    ratio = np.abs(y_pred) / np.abs(D_X)
    print(ratio[:100])
    D_out = ratio * D_X * max
    M_out = D_out[:,::2]+1j*D_out[:,1::2]
    F_out = np.dot(M_out,mel2freq_matrix)

    #ratio[np.isnan(ratio)] = 0.0
    print("shape of F_out:",F_out.shape)
    T = rev_window_fft(F_out,fft_size,step_size)

    # write the result
    Tint = T/np.max(T)*32767
    wavfile.write("Denoise_reconstruction.wav",mix_sr,Tint.astype('int16'))



























