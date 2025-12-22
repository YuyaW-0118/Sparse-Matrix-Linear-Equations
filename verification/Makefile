CXX = icpx
CXXFLAGS = -O3 -std=c++17 -fopenmp -DCUB_MKL
LDFLAGS = -lnuma -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

.PHONY: all clean

all: matrix_stats

matrix_stats: matrix_stats.cpp ../sparse_matrix.h
	$(CXX) $(CXXFLAGS) -o _$@ $< $(LDFLAGS)

clean:
	rm -f matrix_stats
