#!/usr/local/bin/fish

# Some code to generate figures from image sets

set SOS_images s1.png s2.png s3.png
montage $SOS_images -geometry +3+1 subitizing.png
