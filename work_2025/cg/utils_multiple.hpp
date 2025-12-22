#ifndef UTILS_MULTIPLE_HPP
#define UTILS_MULTIPLE_HPP

#include <omp.h>
#include <mkl.h>

// dot product for multiple vectors: result[i] = x_i' * y_i
template <typename ValueT, typename OffsetT>
inline void dot_multiple(OffsetT n, int num_vectors, const ValueT *x, const ValueT *y, ValueT *result)
{
	for (int i = 0; i < num_vectors; i++)
		result[i] = 0.0;

#pragma omp parallel for reduction(+ : result[0 : num_vectors])
	for (OffsetT j = 0; j < n; ++j)
	{
		const ValueT *x_row = x + (long long)j * num_vectors;
		const ValueT *y_row = y + (long long)j * num_vectors;

#pragma omp simd
		for (int i = 0; i < num_vectors; ++i)
			result[i] += x_row[i] * y_row[i];
	}
}

// axpy for multiple vectors: y_i = y_i + a[i]*x_i
template <typename ValueT, typename OffsetT>
inline void axpy_multiple(OffsetT n, int num_vectors, const ValueT *a, const ValueT *x, ValueT *y)
{
#pragma omp parallel for
	for (OffsetT j = 0; j < n; ++j)
	{
		ValueT *y_row = y + (long long)j * num_vectors;
		const ValueT *x_row = x + (long long)j * num_vectors;

#pragma omp simd
		for (int i = 0; i < num_vectors; ++i)
		{
			y_row[i] += a[i] * x_row[i];
		}
	}
}
// p_i = r_i + beta[i] * p_i
template <typename ValueT, typename OffsetT>
inline void update_p_multiple(OffsetT n, int num_vectors, const ValueT *r, const ValueT *beta, ValueT *p)
{
#pragma omp parallel for
	for (OffsetT j = 0; j < n; ++j)
	{
		const ValueT *r_row = r + (long long)j * num_vectors;
		ValueT *p_row = p + (long long)j * num_vectors;

#pragma omp simd
		for (int i = 0; i < num_vectors; ++i)
		{
			p_row[i] = r_row[i] + beta[i] * p_row[i];
		}
	}
}

#endif // UTILS_MULTIPLE_HPP
