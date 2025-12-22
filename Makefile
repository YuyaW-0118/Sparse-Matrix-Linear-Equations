#/******************************************************************************
# * Copyright (c) 2011, Duane Merrill.  All rights reserved.
# * Copyright (c) 2011-2016, NVIDIA CORPORATION.  All rights reserved.
# * 
# * Redistribution and use in source and binary forms, with or without
# * modification, are permitted provided that the following conditions are met:
# *	 * Redistributions of source code must retain the above copyright
# *	   notice, this list of conditions and the following disclaimer.
# *	 * Redistributions in binary form must reproduce the above copyright
# *	   notice, this list of conditions and the following disclaimer in the
# *	   documentation and/or other materials provided with the distribution.
# *	 * Neither the name of the NVIDIA CORPORATION nor the
# *	   names of its contributors may be used to endorse or promote products
# *	   derived from this software without specific prior written permission.
# * 
# * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
# * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *
#******************************************************************************/

#-------------------------------------------------------------------------------
#
# Makefile usage
# 
# CPU:
# make cpu_spmv
#
# GPU:
# make gpu_spmv [sm=<XXX,...>] [verbose=<0|1>] 
#
#-------------------------------------------------------------------------------
 
#-------------------------------------------------------------------------------
# Commandline Options
#-------------------------------------------------------------------------------


# [sm=<XXX,...>] Compute-capability to compile for, e.g., "sm=200,300,350" (SM20 by default).
  
COMMA = ,
ifdef sm
	SM_ARCH = $(subst $(COMMA),-,$(sm))
else 
    SM_ARCH = 350
endif

ifeq (520, $(findstring 520, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_52,code=\"sm_52,compute_52\" 
endif
ifeq (370, $(findstring 370, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_37,code=\"sm_37,compute_37\" 
endif
ifeq (350, $(findstring 350, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_35,code=\"sm_35,compute_35\" 
endif
ifeq (300, $(findstring 300, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_30,code=\"sm_30,compute_30\"
endif


# [verbose=<0|1>] Verbose toolchain output from nvcc option

ifeq ($(verbose), 1)
	NVCCFLAGS += -v
endif



#-------------------------------------------------------------------------------
# Compiler and compilation platform
#-------------------------------------------------------------------------------

CUB_DIR = $(dir $(lastword $(MAKEFILE_LIST)))

NVCC = "$(shell which nvcc)"
ifdef nvccver
    NVCC_VERSION = $(nvccver)
else
    NVCC_VERSION = $(strip $(shell nvcc --version | grep release | sed 's/.*release //' |  sed 's/,.*//'))
endif

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])

# Default flags: verbose kernel properties (regs, smem, cmem, etc.); runtimes for compilation phases 
NVCCFLAGS += $(SM_DEF) -Xptxas -v -Xcudafe -\# 

ifeq (WIN_NT, $(findstring WIN_NT, $(OSUPPER)))
    # For MSVC
    # Disable excess x86 floating point precision that can lead to results being labeled incorrectly
    NVCCFLAGS += -Xcompiler /fp:strict
    # Help the compiler/linker work with huge numbers of kernels on Windows
	NVCCFLAGS += -Xcompiler /bigobj -Xcompiler /Zm500
	CC = cl
	NPPI = -lnppi
	
	# Multithreaded runtime
	NVCCFLAGS += -Xcompiler /MT
	
ifneq ($(force32), 1)
	CUDART_CYG = "$(shell dirname $(NVCC))/../lib/Win32/cudart.lib"
else
	CUDART_CYG = "$(shell dirname $(NVCC))/../lib/x64/cudart.lib"
endif
	CUDART = "$(shell cygpath -w $(CUDART_CYG))"
else
    # For g++
    # Disable excess x86 floating point precision that can lead to results being labeled incorrectly
    NVCCFLAGS += -Xcompiler -ffloat-store
    CC = g++
ifneq ($(force32), 1)
    CUDART = "$(shell dirname $(NVCC))/../lib/libcudart_static.a"
else
    CUDART = "$(shell dirname $(NVCC))/../lib64/libcudart_static.a"
endif
endif


#-------------------------------------------------------------------------------
# Compiler and compilation platform
#-------------------------------------------------------------------------------

# OMP compiler
OMPCC=icpx
OMPCC_FLAGS=-qopenmp -O3 -lrt -fno-alias -march=native -lnuma -qmkl
OMPCC_FLAGS_sse=-qopenmp -O3 -lrt -fno-alias -msse4.2 -lnuma -qmkl
OMPCC_FLAGS_avx512=-qopenmp -O3 -lrt -fno-alias -mavx512f -mavx512dq -lnuma -qmkl
OMPCC_FLAGS_avx=-qopenmp -O3 -lrt -fno-alias -mavx2 -mfma -lnuma -qmkl

# Includes
INC += -I$(CUB_DIR) -I$(CUB_DIR)test 

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])

#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

rwildcard=$(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *,%,$2),$d))

DEPS = 	$(call rwildcard, $(CUB_DIR),*.cuh) \
		$(call rwildcard, $(CUB_DIR),*.h) \
        Makefile

#-------------------------------------------------------------------------------
# make clean
#-------------------------------------------------------------------------------

clean :
	rm -f _gpu_spmv_driver _cpu_spmv_driver _cpu_spmm_driver _cpu_multicg_driver

		
#-------------------------------------------------------------------------------
# make gpu_spmv
#-------------------------------------------------------------------------------

gpu_spmv : gpu_spmv.cu $(DEPS)
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o _gpu_spmv_driver gpu_spmv.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -lcusparse -O3

	
#-------------------------------------------------------------------------------
# make cpu_spmv
#-------------------------------------------------------------------------------

# init: source /opt/intel/oneapi/setvars.sh

cpu_spmv : cpu_spmv.cpp $(DEPS)
	$(OMPCC) $(DEFINES) -DCUB_MKL -o _cpu_spmv_driver cpu_spmv.cpp $(OMPCC_FLAGS_avx512)

cpu_spmm : cpu_spmm_v2.cpp $(DEPS)
	$(OMPCC) $(DEFINES) -DCUB_MKL -o _cpu_spmm_driver cpu_spmm_v2.cpp $(OMPCC_FLAGS_avx)

axpy : axpy.cpp $(DEPS)
	$(OMPCC) $(DEFINES) -DCUB_MKL -o test_axpy axpy.cpp $(OMPCC_FLAGS_avx512)

merge_spmm : merge_spmm_test.cpp $(DEPS)
	$(OMPCC) $(DEFINES) -DCUB_MKL -o merge_spmm merge_spmm_test.cpp $(OMPCC_FLAGS3)

cpu_multicg: cpu_multicg.cpp $(DEPS)
	$(OMPCC) $(DEFINES) -DCUB_MKL -o _cpu_multicg_driver cpu_multicg.cpp $(OMPCC_FLAGS_avx)

cpu_multicg2: cpu_multicg2.cpp $(DEPS)
	$(OMPCC) $(DEFINES) -DCUB_MKL -o _cpu_multicg2_driver cpu_multicg2.cpp $(OMPCC_FLAGS_avx)