# Speech-denoise-Autoencoder

## Data

Use youtube-dl toolkit to download four TED-talk audio and one cafe noise music.

Preprocess and mix the audio with Sox and ffmpeg to get 9 minutes clean and noised wav file.

Transfer wave data from time scale to frequency scale after rfft.

Calculate mel frequency by the freq-to-mel matrix, and these are the input for autoencoder NN model.

## Structure

<img src="https://github.com/bill9800/Speech-denoise-Autoencoder/pic/raw/master/networkstructure.png" width="600">
