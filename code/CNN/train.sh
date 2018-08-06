#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=project/siamese/solver.prototxt 2>&1| tee project/siamese/caffe.log$@
