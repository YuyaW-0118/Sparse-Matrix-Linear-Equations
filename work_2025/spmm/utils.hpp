#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstddef>
#include <iterator>
#include <iostream>

/**
 * Counting iterator
 */
template <
	typename ValueType,
	typename OffsetT = ptrdiff_t>
struct CountingInputIterator
{
	// Required iterator traits
	typedef CountingInputIterator self_type;				   ///< My own type
	typedef OffsetT difference_type;						   ///< Type to express the result of subtracting one iterator from another
	typedef ValueType value_type;							   ///< The type of the element the iterator can point to
	typedef ValueType *pointer;								   ///< The type of a pointer to an element the iterator can point to
	typedef ValueType reference;							   ///< The type of a reference to an element the iterator can point to
	typedef std::random_access_iterator_tag iterator_category; ///< The iterator category

	ValueType val;

	/// Constructor
	inline CountingInputIterator(
		const ValueType &val) ///< Starting value for the iterator instance to report
		: val(val)
	{
	}

	/// Postfix increment
	inline self_type operator++(int)
	{
		self_type retval = *this;
		val++;
		return retval;
	}

	/// Prefix increment
	inline self_type operator++()
	{
		val++;
		return *this;
	}

	/// Indirection
	inline reference operator*() const
	{
		return val;
	}

	/// Addition
	template <typename Distance>
	inline self_type operator+(Distance n) const
	{
		self_type retval(val + n);
		return retval;
	}

	/// Addition assignment
	template <typename Distance>
	inline self_type &operator+=(Distance n)
	{
		val += n;
		return *this;
	}

	/// Subtraction
	template <typename Distance>
	inline self_type operator-(Distance n) const
	{
		self_type retval(val - n);
		return retval;
	}

	/// Subtraction assignment
	template <typename Distance>
	inline self_type &operator-=(Distance n)
	{
		val -= n;
		return *this;
	}

	/// Distance
	inline difference_type operator-(self_type other) const
	{
		return val - other.val;
	}

	/// Array subscript
	template <typename Distance>
	inline reference operator[](Distance n) const
	{
		return val + n;
	}

	/// Structure dereference
	inline pointer operator->()
	{
		return &val;
	}

	/// Equal to
	inline bool operator==(const self_type &rhs)
	{
		return (val == rhs.val);
	}

	/// Not equal to
	inline bool operator!=(const self_type &rhs)
	{
		return (val != rhs.val);
	}

	/// ostream operator
	friend std::ostream &operator<<(std::ostream &os, const self_type &itr)
	{
		os << "[" << itr.val << "]";
		return os;
	}
};

#endif // UTILS_HPP
