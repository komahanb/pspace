include ${HOME}/git/tacs/Makefile.in
include ${HOME}/git/tacs/TACS_Common.mk

PSPACE_INCLUDE = -I${HOME}/git/pspace/cpp/ -I/usr/include/python2.7
PSPACE_LIB = ${HOME}/git/pspace/cpp/libpspace.a

# This is the one rule that is used to compile all the
# source code in TACSUQ
%.o: %.cpp
	${CXX} ${TACS_CC_FLAGS} ${PSPACE_INCLUDE} -c $< -o $*.o
	@echo
	@echo "        --- Compiled $*.cpp successfully ---"
	@echo

OBJS = TACSStochasticElement.o TACSKSFunction.o TACSStochasticFunction.o \
TACSStochasticVarianceFunction.o TACSStochasticFMeanFunction.o \
TACSStochasticFFMeanFunction.o TACSKSStochasticFMeanFunction.o \
TACSKSStochasticFFMeanFunction.o TACSKineticEnergy.o TACSPotentialEnergy.o \
TACSDisplacement.o TACSVelocity.o TACSKSStochasticFunction.o smd.o

library: ${OBJS}
	ar rcs libtacsuq.a ${OBJS}
	mpicxx -shared ${TACS_CC_FLAGS} ${OBJS} -o libtacsuq.so

default: ${OBJS}

debug: TACS_CC_FLAGS=${TACS_DEBUG_CC_FLAGS}
debug: default

complex: TACS_DEF=-DTACS_USE_COMPLEX -DUSE_COMPLEX
complex: default library

complex_debug: TACS_DEF=-DTACS_USE_COMPLEX -DUSE_COMPLEX
complex_debug: debug

clean:
	rm -f *.o libtacsuq.a libtacsuq.so smd ppa vpl fourbar

test: default

test_complex: complex
