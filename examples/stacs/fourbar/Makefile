include ${HOME}/git/tacs/Makefile.in
include ${HOME}/git/tacs/TACS_Common.mk

# PSPACE CORE
PSPACE_INCLUDE = -I${HOME}/git/pspace/cpp/
PSPACE_LIB = ${HOME}/git/pspace/cpp/libpspace.a 

# STACS
STACS_INCLUDE = -I${HOME}/git/pspace/examples/stacs/cpp
STACS_LIB = ${HOME}/git/pspace/examples/stacs/libstacs.a  

# OPTIMIZATION Package
IPOPT_INCLUDE = -I${HOME}/git/ipopt/build/include/coin/
IPOPT_LIB = -L${HOME}/git/ipopt/build/lib -lipopt -llapack -lblas -lm -ldl
IPOPT_LD_FLAGS = -Wl,--rpath -Wl,${HOME}/git/ipopt/build/lib

# This is the one rule that is used to compile all the source code in TACS
%.o: %.cpp
	${CXX} ${TACS_CC_FLAGS} ${PSPACE_INCLUDE} ${STACS_INCLUDE} ${IPOPT_INCLUDE} -c $< -o $*.o
	@echo
	@echo "        --- Compiled $*.cpp successfully ---"
	@echo

OBJS =  deterministic.o projection.o sampling.o

default: ${OBJS}
	${CXX} -o deterministic deterministic.o ${TACS_LD_FLAGS} ${PSPACE_LIB} ${STACS_LIB} 
	${CXX} -o projection projection.o ${TACS_LD_FLAGS} ${PSPACE_LIB} ${STACS_LIB} 
	${CXX} -o sampling sampling.o ${TACS_LD_FLAGS} ${PSPACE_LIB} ${STACS_LIB}

opt: TACS_CC_FLAGS+= -DOPT
opt: ${OBJS} DetOpt.o ProjectionOUU.o SamplingOUU.o
	${CXX} -o detopt deterministic.o DetOpt.o ${TACS_LD_FLAGS} ${PSPACE_LIB} ${STACS_LIB} ${IPOPT_LIB} ${IPOPT_LD_FLAGS}
	${CXX} -o projectionouu projection.o ProjectionOUU.o ${TACS_LD_FLAGS} ${PSPACE_LIB} ${STACS_LIB} ${IPOPT_LIB} ${IPOPT_LD_FLAGS}
	${CXX} -o samplingouu sampling.o SamplingOUU.o ${TACS_LD_FLAGS} ${PSPACE_LIB} ${STACS_LIB} ${IPOPT_LIB} ${IPOPT_LD_FLAGS}

debug: TACS_CC_FLAGS=${TACS_DEBUG_CC_FLAGS}
debug: default

complex: TACS_DEF=-DTACS_USE_COMPLEX -DUSE_COMPLEX
complex: default

complex_debug: TACS_DEF=-DTACS_USE_COMPLEX -DUSE_COMPLEX
complex_debug: debug

clean:
	rm -f *.o deterministic sampling projection

test: default
	./deterministic
	./sampling
	./projection

test_complex: complex
	./deterministic
	./sampling
	./projection
