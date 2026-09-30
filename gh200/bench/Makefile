NH        ?= /software/nvhpc/Linux_aarch64/26.5
CUDA_HOME ?= $(NH)/cuda/13.2
MATH_HOME ?= $(NH)/math_libs/13.2
NVCC      := $(CUDA_HOME)/bin/nvcc
FLAGS     := -O3 -std=c++17 -arch=sm_90a -I$(MATH_HOME)/include
LIBS      := -L$(MATH_HOME)/lib64 -Xlinker -rpath=$(MATH_HOME)/lib64 -lcublas -lcublasLt \
             -L$(CUDA_HOME)/lib64 -Xlinker -rpath=$(CUDA_HOME)/lib64 -lcudart

all: ../bin/gemm_mant_bench ../bin/fp64_kernel_bench

../bin/%: %.cu
	$(NVCC) $(FLAGS) -o $@ $< $(LIBS)
