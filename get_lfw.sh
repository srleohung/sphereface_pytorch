#!/bin/bash
# macOS
# brew install wget

# download lfw
wget -O lfw.tgz http://vis-www.cs.umass.edu/lfw/lfw.tgz

# convert lfw.tgz to lfw.zip
tar zxf lfw.tgz
cd lfw
zip -r ../lfw.zip *
cd ..