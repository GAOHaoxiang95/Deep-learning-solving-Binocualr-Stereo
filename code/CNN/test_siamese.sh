#!/usr/bin/env sh
set -e
./build/tools/caffe test --model=project/siamese/train_test.prototxt -weights=project/siamese/siamese_iter_160000.caffemodel $@
