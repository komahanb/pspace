# Execute the sequence of commands to build the library
echo "compiling probabilistic space [PSPACE]"
cd cpp
make clean
make
echo "compiling stochastic TACS [STACS]"
cd ../
cd examples/stacs/cpp
make clean
make

