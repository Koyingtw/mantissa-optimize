# MI350X application kernels: make -C mi350/apps  →  mi350/bin/{dfmp2,gp_chol,g8_repro}
ROCM ?= /opt/rocm-7.2.3
G8_SRC ?= $(HOME)/koying/sw/rocHPL-amd/third_party/GEMMul8
G8_LIB ?= $(HOME)/koying/sw/gemmul8/lib
HIPCC = $(ROCM)/bin/hipcc -O3 -std=c++17 --offload-arch=gfx950
G8 = -DWITH_GEMMUL8 -I$(G8_SRC)/include -L$(G8_LIB) -Wl,-rpath=$(G8_LIB) -lgemmul8
BIN = ../bin

all: $(BIN)/dfmp2 $(BIN)/gp_chol $(BIN)/g8_repro

$(BIN)/dfmp2: dfmp2.hip
	$(HIPCC) -o $@ $< $(G8) -lhipblas
$(BIN)/gp_chol: gp_chol.hip
	$(HIPCC) -o $@ $< -lrocblas -lrocsolver -lhipblas
$(BIN)/g8_repro: g8_repro.hip
	$(HIPCC) -o $@ $< $(G8) -lhipblas
