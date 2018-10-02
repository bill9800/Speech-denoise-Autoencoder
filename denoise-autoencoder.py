import sys
sys.path.append('lib')
import AVHandler as avh
import AVPreprocess as avp
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from sklearn.model_selection import train_test_split

# option
DOWNLOAD = 0 #download and preprocess the data with little samples (demo)
WAVE_PLOT = 0 # plot original wave, noise wave, mixed wave
INVERSE_CHECK = 0 # check the inverse function of mel
DUMP = 0 # dump wave data to real wav
TRAIN_DENOISE = 0 # train the denoising model with mel freq input and output
DENOISE = 1 # use the pretrained denoise autoencoder

###################################################################
if DOWNLOAD:
    avh.mkdir('sample')
    clean_video_list = ['https://www.youtube.com/watch?v=DCS6t6NUAGQ&t',
                        'https://www.youtube.com/watch?v=gN9dlisaQVM',
                        'https://www.youtube.com/watch?v=c0KYU2j0TM4',
                        'https://www.youtube.com/watch?v=8S0FDjFBj8o']

    for i in range(len(clean_video_list)):
        name = 'clean_' + str(i)
        avh.download('sample',name,clean_video_list[i])
        start_time = 60
        end_time = 180
        avh.cut('sample',name,start_time,end_time)
    avh.conc('sample','clean')

    noise_link = 'https://www.youtube.com/watch?v=BOdLmxy06H0'
    name = 'noise'
    avh.download('sample', name, noise_link)
    noise_s_time = 60
    noise_e_time = 540
    avh.cut('sample',name,noise_s_time,noise_e_time)

    avh.mix('sample','mix','clean','noise',0,480)

####################################################################

# mix clean audio and noise audio
with open('sample/clean.wav', 'rb') as f:
    clean_data, clean_sr = librosa.load('sample/clean.wav', sr=None)  # time series data,sample rate
with open('sample/noise.wav', 'rb') as f:
    noise_data, noise_sr = librosa.load('sample/noise.wav', sr=None)  # time series data,sample rate


# normalize expand the noise
noise_max = np.max(noise_data)
expand_rate = 1/noise_max
noise_data = noise_data*expand_rate

assert clean_sr == noise_sr
mix_data = clean_data*0.8 + noise_data*0.2
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
step_size = fft_size // 3 # distance to slide along the window

# fequency to mel parameter #
n_mels = 40 # number of mel frequency
start_freq = 0.0
end_freq = 8000.0

# implement mel to time (just to check inverse function)
if INVERSE_CHECK:
    M = avp.time_to_mel(mix_data,mix_sr,fft_size,n_mels,step_size)
    T = avp.mel_to_time(M,mix_sr,fft_size,n_mels,step_size)
    plt.figure()
    plt.plot(T)
    plt.show()
    Tint = T/max(T)*32767
    wavfile.write("mix/test.wav",mix_sr,Tint.astype('int16'))

###################################################################
#split data

mel_mix_data = avp.time_to_mel(mix_data,mix_sr,fft_size,n_mels,step_size)
D_X = avp.real_imag_expand(mel_mix_data)

mel_clean_data = avp.time_to_mel(clean_data,clean_sr,fft_size,n_mels,step_size,fmax=8000)
D_y = avp.real_imag_expand(mel_clean_data)

# separate data to train test sets
D_X_train = avp.min_max_norm(D_X[:int(D_X.shape[0]*0.9),:])
D_y_train = D_y[:int(D_y.shape[0]*0.9),:] / D_X[:int(D_X.shape[0]*0.9),:]
G_max = np.max(D_y_train)
D_y_train = D_y_train/G_max

X_test = avp.min_max_norm(D_X[int(D_X.shape[0]*0.9):,:])
y_test = D_y[int(D_y.shape[0]*0.9):,:] / D_X[int(D_X.shape[0]*0.9):,:]
y_test = y_test/G_max

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

    n_hidden1 = 2049
    n_hidden2 = 500
    n_hidden3 = 180

    InputLayer1 = Input(shape=(n_input_dim,), name="InputLayer")
    InputLayer2 = BatchNormalization(axis=1, momentum=0.6)(InputLayer1)

    HiddenLayer1_1 = Dense(n_hidden1, name="H1", activation='relu', kernel_initializer=he_normal(seed=27))(InputLayer2)
    HiddenLayer1_2 = BatchNormalization(axis=1, momentum=0.6)(HiddenLayer1_1)
    HiddenLayer1_3 = Dropout(0.1)(HiddenLayer1_2)

    HiddenLayer2_1 = Dense(n_hidden2, name="H2", activation='relu', kernel_initializer=he_normal(seed=42))(HiddenLayer1_3)
    HiddenLayer2_2 = BatchNormalization(axis=1, momentum=0.6)(HiddenLayer2_1)

    HiddenLayer3_1 = Dense(n_hidden3, name="H3", activation='relu', kernel_initializer=he_normal(seed=65))(HiddenLayer2_2)
    HiddenLayer3_2 = BatchNormalization(axis=1, momentum=0.6)(HiddenLayer3_1)

    HiddenLayer2__1 = Dense(n_hidden2, name="H2_R", activation='relu', kernel_initializer=he_normal(seed=42))(HiddenLayer3_2)
    HiddenLayer2__2 = BatchNormalization(axis=1, momentum=0.6)(HiddenLayer2__1)

    HiddenLayer1__1 = Dense(n_hidden1, name="H1_R", activation='relu', kernel_initializer=he_normal(seed=27))(HiddenLayer2__2)
    HiddenLayer1__2 = BatchNormalization(axis=1, momentum=0.6)(HiddenLayer1__1)
    HiddenLayer1__3 = Dropout(0.1)(HiddenLayer1__2)

    OutputLayer = Dense(n_output_dim, name="OutputLayer", kernel_initializer=he_normal(seed=62))(HiddenLayer1__3)

    model = Model(inputs=[InputLayer1], outputs=[OutputLayer])
    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0001, amsgrad=False)
    # loss = p_loss(OutputLayer,K.placeholder())
    model.compile(loss='mse', optimizer=opt)

    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    model.summary()

    tensorboard = TensorBoard(log_dir="./logs", histogram_freq=0, write_graph=True, write_images=True)
    # fit the model
    hist = model.fit(X_train, y_train, batch_size=512, epochs=100, verbose=1, validation_data=([X_val], [y_val]),
                     callbacks=[tensorboard])

    plt.figure(figsize=(10, 8))
    plt.plot(hist.history['loss'], label='Loss')
    plt.plot(hist.history['val_loss'], label='Val_Loss')
    plt.legend(loc='best')
    plt.title('Training Loss and Validation Loss')
    plt.show()

    results = model.evaluate(X_test, y_test, batch_size=len(y_test))
    print('Test loss:%3f' % results)

    # serialize model to JSON
    model_json = model.to_json()
    avh.mkdir('model')
    with open("model/model.json", 'w') as f:
        f.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model/model.h5")
    print("Saved model to disk")

if DENOISE:
    # load josn and create model
    with open('model/model.json','r') as f:
        loaded_model_json = f.read()
    denoise_model = model_from_json(loaded_model_json)
    denoise_model.load_weights("model/model.h5")
    print("Loaded model from disk")

    gain = denoise_model.predict(D_X) * G_max
    M_gain = gain[:,::2]+1j*gain[:,1::2]
    F_gain = avp.mel2freq(M_gain,mix_sr,fft_size,n_mels)

    F = F_gain * avp.stft(mix_data,fft_size,step_size)
    #ratio[np.isnan(ratio)] = 0.0
    print("shape of F_out:",F.shape)
    T = avp.istft(F,fft_size,step_size)

    # write the result
    Tint = T/np.max(T)*32767
    wavfile.write("Denoise_reconstruction.wav",mix_sr,Tint.astype('int16'))



























