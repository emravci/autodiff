#pragma once

#include <array>

#include "../utility/arithmetic.hpp"

namespace linalg
{
    template<class Type, std::size_t N> class Vector;
    template<template<typename> class OperatorType, class Type, std::size_t N, class ...Args>
    Vector<Type, N> elementWise(const Vector<Type, N>&, Args&&...);
    template<template<typename> class OperatorType, class Type, std::size_t N>
    Vector<Type, N> broadcast(const Type, const Vector<Type, N>&);
}

template<class Type, std::size_t N>
class linalg::Vector
{
    public:
    using SizeType = std::size_t;
    Vector() = default;
    Vector(const Type (&x)[N]) : elements_{std::to_array(x)} {}
    Type& operator[](SizeType i) { return elements_[i]; }
    const Type& operator[](SizeType i) const { return elements_[i]; }
    private:
    std::array<Type, N> elements_{};
};

template<template<typename> class OperatorType, class Type, std::size_t N, class ...Args>
linalg::Vector<Type, N> linalg::elementWise(const Vector<Type, N>& that, Args&& ...args)
{
    using OperandType = linalg::Vector<Type, N>;
    using SizeType = typename OperandType::SizeType;

    static_assert(sizeof...(Args) < 2, "this function could have 1 or 2 arguments at most");
    static_assert((std::is_same_v<OperandType, std::decay_t<Args>> && ...), "both of the arguments should have the same type");    

    OperatorType<Type> op;
    OperandType result;
    for(SizeType i = 0; i < N; ++i) { result[i] = op(that[i], args[i]...); }
    return result;
}

template<class Type, std::size_t N>
linalg::Vector<Type, N> operator+(const linalg::Vector<Type, N>& lhs, const linalg::Vector<Type, N>& rhs)
{
    return linalg::elementWise<utility::Add>(lhs, rhs);
}

template<class Type, std::size_t N>
linalg::Vector<Type, N> operator-(const linalg::Vector<Type, N>& that)
{
    return linalg::elementWise<utility::Negate>(that);
}

template<class Type, std::size_t N>
linalg::Vector<Type, N> operator-(const linalg::Vector<Type, N>& lhs, const linalg::Vector<Type, N>& rhs)
{
    return linalg::elementWise<utility::Subtract>(lhs, rhs);
}

template<template<typename> class OperatorType, class Type, std::size_t N>
linalg::Vector<Type, N> linalg::broadcast(const Type lhs, const linalg::Vector<Type, N>& rhs)
{
    using ResultType = linalg::Vector<Type, N>;
    using SizeType = typename ResultType::SizeType;

    OperatorType<Type> op;
    ResultType result;
    for(SizeType i = 0; i < N; ++i) { result[i] = op(lhs, rhs[i]); }
    return result;
}

template<class Type, std::size_t N>
linalg::Vector<Type, N> operator*(const Type lhs, const linalg::Vector<Type, N>& rhs)
{
    return linalg::broadcast<utility::Multiply>(lhs, rhs);
}

template<class Type, std::size_t N>
std::ostream& operator<<(std::ostream& out, const linalg::Vector<Type, N>& that)
{
	using SizeType = typename linalg::Vector<Type, N>::SizeType;
    out << "[ ";
    for(SizeType i = 0; i < N; ++i) { out << that[i] << " "; }
    out << "]";
	return out;
}
