#!/bin/bash

# API call to get JPEG images from ZM.
# This a mini tool used to get some real life image without having
# to removed the URL all the time.
wget "http://localhost:8080/zm/cgi-bin/zms?mode=single&monitor=1"  -O foo.jpg
