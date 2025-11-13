CXX            := mpicxx
BUILD          ?= release
USE_COMPLEX    ?= 0

SOURCES := \
	ArrayList.cpp \
	OrthogonalPolynomials.cpp \
	GaussianQuadrature.cpp \
	AbstractParameter.cpp \
	NormalParameter.cpp \
	UniformParameter.cpp \
	ExponentialParameter.cpp \
	ParameterFactory.cpp \
	BasisHelper.cpp \
	QuadratureHelper.cpp \
	ParameterContainer.cpp

OBJECTS := $(SOURCES:.cpp=.o)

BASE_FLAGS := -fPIC -I../include -funroll-loops

ifeq ($(BUILD),debug)
OPT_FLAGS := -O0 -g
else
OPT_FLAGS := -O3
endif

CXXFLAGS := $(BASE_FLAGS) $(OPT_FLAGS)

ifeq ($(USE_COMPLEX),1)
CXXFLAGS += -DUSE_COMPLEX
endif

.PHONY: default clean

default: libpspace.so libpspace.a a.out

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

libpspace.a: $(OBJECTS)
	ar rcs $@ $^

libpspace.so: $(OBJECTS)
	$(CXX) -shared $(CXXFLAGS) $^ -o $@

a.out: libpspace.a main.cpp
	$(CXX) $(CXXFLAGS) -I. -L. main.cpp -o $@ -lpspace

clean:
	rm -f *.so *.o a.out
