include ${HOME}/git/tacs/Makefile.in
include ${HOME}/git/tacs/TACS_Common.mk

PSPACE_INCLUDE = -I${HOME}/git/pspace/cpp/
PSPACE_LIB = ${HOME}/git/pspace/cpp/libpspace.a

PSPACE_TACS_INCLUDE = -I${HOME}/git/pspace/examples/tacs/
PSPACE_TACS_LIB = ${HOME}/git/pspace/examples/tacs/libtacsuq.a

# This is the one rule that is used to compile all the
# source code in TACS
%.o: %.cpp
	${CXX} ${TACS_CC_FLAGS} ${PSPACE_INCLUDE} ${PSPACE_TACS_INCLUDE} -c $< -o $*.o
	@echo
	@echo "        --- Compiled $*.cpp successfully ---"
	@echo

OBJS =  smd.o deterministic.o sampling.o projection.o

default: ${OBJS}
	${CXX} -o deterministic smd.o deterministic.o ${TACS_LD_FLAGS} ${PSPACE_LIB} ${PSPACE_TACS_LIB}
	${CXX} -o sampling smd.o sampling.o ${TACS_LD_FLAGS} ${PSPACE_LIB} ${PSPACE_TACS_LIB}
	${CXX} -o projection smd.o projection.o ${TACS_LD_FLAGS} ${PSPACE_LIB} ${PSPACE_TACS_LIB}

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
