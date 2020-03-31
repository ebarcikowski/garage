#!/bin/bash

# API call to get JPEG images from ZM.
wget "http://localhost:8080/zm/cgi-bin/zms?mode=single&monitor=1"  -O foo.jpg
