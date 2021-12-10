CUDA_PATH ?= /usr/local/cuda-11.0
HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)
NVCCFLAGS   :=

ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
endif

ALL_CCFLAGS := $(NVCCFLAGS) -std=c++11
ALL_LDFLAGS := $(ALL_CCFLAGS)
INCLUDES  := -I../../common/inc -I./ -I./include
SMS ?= 70

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

TARGETS := cpu_implementation gpu_implementation1 gpu_implementation2 gpu_implementation3 gpu_implementation4 gpu_implementation5
OBJS := cpu_implementation.o kernels.o gpu_implementation1.o gpu_implementation2.o gpu_implementation3.o gpu_implementation4.o gpu_implementation5.o utils.o
DEPS := kernels.o utils.o

all: build

build: $(TARGETS)

%.o:src/%.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

%: %.o $(DEPS)
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+

clean:
	rm -f $(TARGETS) $(OBJS)
