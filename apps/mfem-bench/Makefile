# 用法: make MFEM_BUILD=/path/to/mfem/build-cuda OUT=ex1_bench.build-cuda
MFEM_BUILD ?= /nfs/home/koying/mantissa-optimize/apps/mfem/build-cuda
OUT ?= ex1_bench
MFEM_DIR := /nfs/home/koying/mantissa-optimize/apps/mfem
MFEM_BUILD_DIR := $(MFEM_BUILD)
include $(MFEM_BUILD)/config/config.mk

$(OUT): ex1_bench.cpp $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) -lineinfo ex1_bench.cpp -o $(OUT) $(MFEM_LIBS)
