include ${HOME}/git/tacs/Makefile.in
include ${HOME}/git/tacs/TACS_Common.mk

OBJS = smd.o ppa.o

default: ${OBJS}
	${CXX} -o smd smd.o ${TACS_LD_FLAGS}
	${CXX} -o ppa ppa.o ${TACS_LD_FLAGS}

debug: TACS_CC_FLAGS=${TACS_DEBUG_CC_FLAGS}
debug: default

complex: TACS_DEF="-DTACS_USE_COMPLEX"
complex: default

complex_debug: TACS_DEF="-DTACS_USE_COMPLEX"
complex_debug: debug

clean:
	rm -f *.o smd ppa 

test: default
	./smd
	./ppa

test_complex: complex
	./smd
	./ppa
