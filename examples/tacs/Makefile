include ${HOME}/git/tacs/Makefile.in
include ${HOME}/git/tacs/TACS_Common.mk

PSPACE_INCLUDE = -I${HOME}/git/pspace/cpp/
PSPACE_LIB = ${HOME}/git/pspace/cpp/libpspace.a

# This is the one rule that is used to compile all the
# source code in TACS
%.o: %.cpp
	${CXX} ${TACS_CC_FLAGS} ${PSPACE_INCLUDE} -c $< -o $*.o
	@echo
	@echo "        --- Compiled $*.cpp successfully ---"
	@echo

OBJS = TACSStochasticElement.o TACSKSFunction.o TACSStochasticFunction.o TACSStochasticVarianceFunction.o TACSStochasticFMeanFunction.o TACSKSStochasticFMeanBeamFunction.o TACSStochasticFFMeanFunction.o TACSKSStochasticFMeanFunction.o TACSKSStochasticFFMeanFunction.o TACSKineticEnergy.o TACSPotentialEnergy.o TACSDisplacement.o TACSVelocity.o TACSKSStochasticFunction.o 

library: ${OBJS}
	ar rcs libtacsuq.a  TACSStochasticElement.o TACSKSFunction.o TACSStochasticFunction.o TACSStochasticVarianceFunction.o TACSStochasticFMeanFunction.o TACSKSStochasticFMeanBeamFunction.o TACSStochasticFFMeanFunction.o TACSKSStochasticFMeanFunction.o TACSKSStochasticFFMeanFunction.o TACSKineticEnergy.o TACSPotentialEnergy.o TACSDisplacement.o TACSVelocity.o TACSKSStochasticFunction.o

default: ${OBJS}

debug: TACS_CC_FLAGS=${TACS_DEBUG_CC_FLAGS}
debug: default

complex: TACS_DEF=-DTACS_USE_COMPLEX -DUSE_COMPLEX
complex: default library

complex_debug: TACS_DEF=-DTACS_USE_COMPLEX -DUSE_COMPLEX
complex_debug: debug

clean:
	rm -f *.o smd ppa vpl fourbar

test: default

test_complex: complex
