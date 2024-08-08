#pragma once

#include <cassert>
#include <cmath>

#include "../linalg/Vector.hpp"

// in forward mode derivatives retrieved from wolfram using, for example, d/dx(sin(f(x)))	= cos(f(x)) * f'(x)
// in reverse mode derivatives retrieved from wolfram using, for example, d/dx(sin(f(x)))	= cos(f(x))

namespace autodiff 
{
    template<class Type, std::size_t N> class Forward; 
    template<class Type, std::size_t N> Forward<Type, N> sin(const Forward<Type, N>&);
    template<class Type, std::size_t N> Forward<Type, N> cos(const Forward<Type, N>&);
    template<class Type, std::size_t N> Forward<Type, N> tan(const Forward<Type, N>&);
    template<class Type, std::size_t N> Forward<Type, N> exp(const Forward<Type, N>&);
    template<class Type, std::size_t N> Forward<Type, N> log(const Forward<Type, N>&);
    template<class Type, std::size_t N> Forward<Type, N> pow(const Forward<Type, N>&, const Forward<Type, N>&);
}

template<class Type, std::size_t N>
class autodiff::Forward
{   
    public:
    using SelfType = Forward<Type, N>;
    using GradientType = linalg::Vector<Type, N>;
    using SizeType = std::size_t;
    Forward() : value_(0) {}
    Forward(Type value) : value_{value} {}
    Forward(Type value, SizeType i) : value_{value}, index_{i}
    {
        assert(i < N && "index cannot be bigger than or equal to number of variables");
        gradient_[i] = Type(1);
    }
    Forward(Type value, GradientType gradient) : value_{value}, gradient_{std::move(gradient)} {}
    Type value() const { return value_; }
    SizeType index() const { return index_; }
    const GradientType& gradient() const { return gradient_; }
    SelfType conjugate() const { return Forward{value_, -gradient_}; }
    private:
    Type value_;
    GradientType gradient_;
    SizeType index_{N};
};

template<class Type, std::size_t N>
autodiff::Forward<Type, N> operator+(const autodiff::Forward<Type, N>& lhs, const autodiff::Forward<Type, N>& rhs)
{
    return autodiff::Forward<Type, N>(lhs.value() + rhs.value(), lhs.gradient() + rhs.gradient());
}

template<class Type, std::size_t N>
autodiff::Forward<Type, N> operator-(const autodiff::Forward<Type, N>& that)
{
    return autodiff::Forward<Type, N>(-that.value(), -that.gradient());
}

template<class Type, std::size_t N>
autodiff::Forward<Type, N> operator-(const autodiff::Forward<Type, N>& lhs, const autodiff::Forward<Type, N>& rhs)
{
    return autodiff::Forward<Type, N>(lhs.value() - rhs.value(), lhs.gradient() - rhs.gradient());
}

template<class Type, std::size_t N>
autodiff::Forward<Type, N> operator*(const autodiff::Forward<Type, N>& lhs, const autodiff::Forward<Type, N>& rhs)
{
    return autodiff::Forward<Type, N>(lhs.value() * rhs.value(), rhs.value() * lhs.gradient() + lhs.value() * rhs.gradient());
}

template<class Type, std::size_t N>
autodiff::Forward<Type, N> operator/(const autodiff::Forward<Type, N>& lhs, const autodiff::Forward<Type, N>& rhs)
{
    return autodiff::Forward<Type, N>(Type(1) / rhs.value() / rhs.value()) * lhs * rhs.conjugate();
}

template<class Type, std::size_t N>
autodiff::Forward<Type, N> autodiff::sin(const autodiff::Forward<Type, N>& that)
{
    using std::sin, std::cos;
    return autodiff::Forward<Type, N>(sin(that.value()), cos(that.value()) * that.gradient());
}

template<class Type, std::size_t N>
autodiff::Forward<Type, N> autodiff::cos(const autodiff::Forward<Type, N>& that)
{
    using std::cos, std::sin;
    return autodiff::Forward<Type, N>(cos(that.value()), -sin(that.value()) * that.gradient());
}

template<class Type, std::size_t N>
autodiff::Forward<Type, N> autodiff::tan(const autodiff::Forward<Type, N>& that)
{
    using std::tan, std::pow, std::cos;
    return autodiff::Forward<Type, N>(tan(that.value()), (Type(1) / cos(that.value()) / cos(that.value())) * that.gradient());
}

template<class Type, std::size_t N>
autodiff::Forward<Type, N> autodiff::exp(const autodiff::Forward<Type, N>& that)
{
    using std::exp;
    return autodiff::Forward<Type, N>(exp(that.value()), exp(that.value()) * that.gradient());
}

template<class Type, std::size_t N>
autodiff::Forward<Type, N> autodiff::log(const autodiff::Forward<Type, N>& that)
{
    using std::log;
    return autodiff::Forward<Type, N>(log(that.value()), (Type(1) / that.value()) * that.gradient());
}

template<class Type, std::size_t N>
autodiff::Forward<Type, N> autodiff::pow(const Forward<Type, N>& x, const autodiff::Forward<Type, N>& y)
{
    using std::pow, std::log;
    return autodiff::Forward<Type, N>(
        pow(x.value(), y.value()), 
        pow(x.value(), y.value() - Type(1)) * (y.value() * x.gradient() + x.value() * log(x.value()) * y.gradient()));
}

template<class Type, std::size_t N>
std::ostream& operator<<(std::ostream& out, const autodiff::Forward<Type, N>& that)
{
	using SizeType = typename linalg::Vector<Type, N>::SizeType;
    out << that.value() << " [ ";
    for(SizeType i = 0; i < N; ++i) { out << that.gradient()[i] << " "; }
    out << "]";
	return out;
}
