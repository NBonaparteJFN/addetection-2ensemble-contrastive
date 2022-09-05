#!/bin/bash

for entry in /c/Users/bonap/Documents/ADReSS-IS2020-data/test/Normalised_audio-chunks/*; #change this to the path of the directory that contains the audio files then put this /* to loop on all the files
do
  fn=${entry%.*}

  # change the first part of this path to the path of the opensmile directory you downloaded during installation
  SMILExtract -C /c/Users/bonap/Documents/opensmile-3.0-win-x64/config/is09-13/IS13_ComParE.conf -I "$entry" -O "test_normalizedaudio/compare_${fn##*/}.csv"
done
