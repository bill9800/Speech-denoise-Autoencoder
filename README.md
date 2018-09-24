# Speech-denoise-Autoencoder

## Data

Use youtube-dl toolkit to download four TED-talk audio and one cafe noise music.

Preprocess and mix the audio with Sox and ffmpeg to get 9 minutes clean and noised wav file.

Transfer wave data from time scale to frequency scale after rfft.

Calculate mel frequency by the freq-to-mel matrix, and these are the input for autoencoder NN model.

## Structure

This model train mel freqency of the noised data as input and the gain as output.

<img src="https://github.com/bill9800/Speech-denoise-Autoencoder/raw/master/pic/networkstructure.png" width="300">

## Prediction

Input the noised test data to this model, the loss is around 2*1e-5.

Reconstruct the data and output the wav file from the gain method.










