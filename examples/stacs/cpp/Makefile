include ${HOME}/git/tacs/Makefile.in
include ${HOME}/git/tacs/TACS_Common.mk

include ${HOME}/git/tmr/Makefile.in
include ${HOME}/git/tmr/TMR_Common.mk

PSPACE_INCLUDE = -I${HOME}/git/pspace/cpp/ -I/usr/include/python2.7

# This is the one rule that is used to compile
%.o: %.cpp
	${CXX} ${TACS_CC_FLAGS} ${TMR_CC_FLAGS} ${PSPACE_INCLUDE} -c $< -o $*.o
	@echo
	@echo "        --- Compiled $*.cpp successfully ---"
	@echo

OBJS = TACSStochasticElement.o TACSKSFunction.o TACSStochasticFunction.o \
TACSStochasticVarianceFunction.o TACSStochasticFMeanFunction.o \
TACSStochasticFFMeanFunction.o TACSKSStochasticFMeanFunction.o \
TACSKSStochasticFFMeanFunction.o TACSKineticEnergy.o TACSPotentialEnergy.o \
TACSDisplacement.o TACSVelocity.o TACSKSStochasticFunction.o smd.o \
TACSMutableElement3D.o

library: ${OBJS}
	ar rcs libstacs.a ${OBJS}
	mpicxx -shared ${TACS_CC_FLAGS} ${TMR_CC_FLAGS} ${OBJS} -o libstacs.so

default: ${OBJS}

debug: TACS_CC_FLAGS=${TACS_DEBUG_CC_FLAGS}
debug: default library

complex: TACS_DEF=-DTACS_USE_COMPLEX -DUSE_COMPLEX
complex: default library

complex_debug: TACS_DEF=-DTACS_USE_COMPLEX -DUSE_COMPLEX
complex_debug: debug

clean:
	rm -f *.o libstacs.a libstacs.so

test: default

test_complex: complex
