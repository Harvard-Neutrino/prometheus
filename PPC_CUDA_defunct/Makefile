ifndef arch
arch	=	52
endif

nvcc	=	nvcc ppc.cu -Xptxas=-v -arch=sm_$(arch) \
		-O2 --use_fast_math --compiler-options=-O2,--fast-math

gcpp	=	$(CXX) ppc.cxx -O2 --fast-math

mlib	=	-fPIC -DXLIB -c -o ppc.o && $(CC) -shared \
		-fPIC -Wl,-soname,xppc ppc.o -o libxppc.so

warn	=	2>&1 | grep -v assuming

all:
	@echo "	make gpu:  compile the ppc  executable for GPU"
	@echo "	make cpu:  compile the ppc  executable for CPU"
	@echo "	make glib: compile the libxppc library for GPU"
	@echo "	make clib: compile the libxppc library for CPU"

go:
	$(nvcc) -o ppc.o -DXLIB -c $(warn)

gpu:
	$(nvcc) -o ppc $(warn)

glib:
	$(nvcc),$(mlib)

link:
	ln -s ppc.cu ppc.cxx || true
	ln -s pro.cu pro.cxx || true

co:	link
	$(gcpp) -o ppc.o -DXLIB -c

cpu:	link
	$(gcpp) -o ppc

clib:	link
	$(gcpp) $(mlib)

clib_osx:	link
	$(gcpp) -fPIC -DXLIB -dynamic -dynamiclib -o libxppc.dylib

clean:
	rm ppc.o ppc libxppc.so ppc.cxx pro.cxx || true
