#ifndef TYPES_HPP
#define TYPES_HPP

struct int2
{
	int x;
	int y;
};

// Enum to specify which SpMM kernel to use
enum SpmmKernel
{
	SIMPLE,
	MERGE,
	NONZERO_SPLIT
};
#endif // TYPES_HPP
