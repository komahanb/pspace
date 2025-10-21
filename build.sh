#!/usr/bin/env bash

# Execute the sequence of commands to build the library
set -euo pipefail

BUILD_MODE="release"
USE_COMPLEX=0

for arg in "$@"; do
  case "$arg" in
    debug)
      BUILD_MODE="debug"
      ;;
    complex)
      USE_COMPLEX=1
      ;;
    complex_debug)
      BUILD_MODE="debug"
      USE_COMPLEX=1
      ;;
    release)
      BUILD_MODE="release"
      ;;
    *)
      echo "Unknown build option: ${arg}" >&2
      exit 1
      ;;
  esac
done

echo "compiling probabilistic space [PSPACE] (build=${BUILD_MODE}, complex=${USE_COMPLEX})"

pushd src/cpp >/dev/null
make clean BUILD="${BUILD_MODE}" USE_COMPLEX="${USE_COMPLEX}"
make BUILD="${BUILD_MODE}" USE_COMPLEX="${USE_COMPLEX}"
popd >/dev/null

mkdir -p lib
rm -f lib/*.so lib/*.a
cp src/cpp/libpspace.so lib/
cp src/cpp/libpspace.a lib/
