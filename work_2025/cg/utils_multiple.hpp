#ifndef UTILS_MULTIPLE_HPP
#define UTILS_MULTIPLE_HPP

#include <omp.h>
#include <mkl.h>

// dot product for multiple vectors: result[i] = x_i' * y_i
template <typename ValueT, typename OffsetT>
inline void dot_multiple(OffsetT n, int num_vectors, const ValueT *x, const ValueT *y, ValueT *result)
{
#pragma omp parallel for
	for (int i = 0; i < num_vectors; ++i)
	{
		ValueT local_result = 0.0;
		const ValueT *x_col = x + (long long)i * n;
		const ValueT *y_col = y + (long long)i * n;
		for (OffsetT j = 0; j < n; ++j)
		{
			local_result += x_col[j] * y_col[j];
		}
		result[i] = local_result;
	}
}

// axpy for multiple vectors: y_i = y_i + a[i]*x_i
template <typename ValueT, typename OffsetT>
inline void axpy_multiple(OffsetT n, int num_vectors, const ValueT *a, const ValueT *x, ValueT *y)
{
#pragma omp parallel for
	for (int i = 0; i < num_vectors; ++i)
	{
		ValueT scalar_a = a[i];
		ValueT *y_col = y + (long long)i * n;
		const ValueT *x_col = x + (long long)i * n;
		for (OffsetT j = 0; j < n; ++j)
		{
			y_col[j] += scalar_a * x_col[j];
		}
	}
}

// p_i = r_i + beta[i] * p_i
template <typename ValueT, typename OffsetT>
inline void update_p_multiple(OffsetT n, int num_vectors, const ValueT *r, const ValueT *beta, ValueT *p)
{
#pragma omp parallel for
	for (int i = 0; i < num_vectors; ++i)
	{
		ValueT scalar_beta = beta[i];
		const ValueT *r_col = r + (long long)i * n;
		ValueT *p_col = p + (long long)i * n;
		for (OffsetT j = 0; j < n; ++j)
		{
			p_col[j] = r_col[j] + scalar_beta * p_col[j];
		}
	}
}

#endif // UTILS_MULTIPLE_HPP
