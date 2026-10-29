ROCM  ?= /opt/rocm-7.2.3
HIPCC := $(ROCM)/bin/hipcc
FLAGS := -O3 -std=c++17 --offload-arch=gfx950 -D__HIP_PLATFORM_AMD__
LIBS  := -L$(ROCM)/lib -Wl,-rpath=$(ROCM)/lib -lhipblas -lhipblaslt

all: ../bin/gemm_mant_bench

../bin/%: %.hip
	@mkdir -p ../bin
	$(HIPCC) $(FLAGS) -o $@ $< $(LIBS)
