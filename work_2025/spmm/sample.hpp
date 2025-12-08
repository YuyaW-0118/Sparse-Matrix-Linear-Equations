#ifndef SAMPLE_HPP
#define SAMPLE_HPP

#include "../../sparse_matrix.h"

//---------------------------------------------------------------------
// SpMV verification
//---------------------------------------------------------------------

// Compute reference SpMV y = Ax
template <
	typename ValueT,
	typename OffsetT>
void SpmvGold(
	CsrMatrix<ValueT, OffsetT> &a,
	ValueT *vector_x,
	ValueT *vector_y_in,
	ValueT *vector_y_out,
	ValueT alpha,
	ValueT beta)
{
	for (OffsetT row = 0; row < a.num_rows; ++row)
	{
		ValueT partial = beta * vector_y_in[row];
		for (
			OffsetT offset = a.row_offsets[row];
			offset < a.row_offsets[row + 1];
			++offset)
		{
			partial += alpha * a.values[offset] * vector_x[a.column_indices[offset]];
		}
		vector_y_out[row] = partial;
	}
}

#endif // SAMPLE_HPP
