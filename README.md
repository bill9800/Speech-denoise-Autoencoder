# Speech-denoising-Autoencoder

Speech denoising systems usually enhance only the magnitude spectrum while leaving the phase spectrum. This system try to improve the performance of denoising system based on denoising autoencoder neural network. The estimation of clean audio is computed by complex ideal ratio mask to enhance the phase information. 

## Structure

Input : audio data on mel-frequency domain

Output: complex ratio mask (cRM)[1]

This model built in linear shape (2049-500-180) without weight lock[2].

## Source

[youtube-dl](http://rg3.github.io/youtube-dl/) : a command-line program to download videos from YouTube.com and a few more sites

[SoX](http://sox.sourceforge.net/) : a cross-platform command line utility to convert various formats of audio files in to other formats

[FFmpeg](https://www.ffmpeg.org/) : a complete, cross-platform solution to record, convert and stream audio and video

[librosa](https://librosa.github.io/librosa/) : python package for music and audio analysis

## Reference

[1] [Complex Ratio Masking for Monaural Speech Separation, D.Williamson, IEEE/ACM TRANSACTIONS ON AUDIO, SPEECH, AND LANGUAGE PROCESSING, VOL. 24, NO. 3, MARCH 2016](https://ieeexplore.ieee.org/document/7364200/)

[2] [Speech Synthesis with Deep Denoising Autoencoder, Zhenzhou Wu](http://gram.cs.mcgill.ca/theses/wu-15-speech.pdf)







