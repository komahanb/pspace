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

OBJS = TACSStochasticElement.o smd.o vpl.o ppa.o fourbar.o

default: ${OBJS}
	${CXX} -o smd TACSStochasticElement.o smd.o ${TACS_LD_FLAGS} ${PSPACE_LIB}
	${CXX} -o vpl TACSStochasticElement.o vpl.o ${TACS_LD_FLAGS} ${PSPACE_LIB}
	${CXX} -o ppa TACSStochasticElement.o ppa.o ${TACS_LD_FLAGS} ${PSPACE_LIB}
	${CXX} -o fourbar TACSStochasticElement.o fourbar.o ${TACS_LD_FLAGS} ${PSPACE_LIB}

debug: TACS_CC_FLAGS=${TACS_DEBUG_CC_FLAGS}
debug: default

complex: TACS_DEF="-DTACS_USE_COMPLEX"
complex: default

complex_debug: TACS_DEF="-DTACS_USE_COMPLEX"
complex_debug: debug

clean:
	rm -f *.o smd ppa vpl fourbar

test: default
	./smd
	./ppa
	./vpl
	./fourbar

test_complex: complex
	./smd
	./ppa
	./vpl
	./fourbar
