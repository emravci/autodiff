#pragma once

#include <array>

#include "../linalg/Vector.hpp"
#include "../linalg/Matrix.hpp"
#include "../linalg/ColumnProxy.hpp"

#include "Forward.hpp"
#include "Reverse.hpp"
#include "typedefs.hpp"

namespace autodiff
{
    template<class Type, std::size_t N>
    linalg::Vector<Type, N> gradient(const Forward<Type, N>&);
    template<class Type, class ...Args>
    linalg::Vector<Type, sizeof...(Args)> gradient(const Reverse<Type>&, Args&&...);
    template<class Type, std::size_t N>
    linalg::Vector<Type, N> gradient(const ForwardOnForward<Type, N>&);
    template<class Type, std::size_t N>
    linalg::Vector<Type, N> gradient(const ReverseOnForward<Type, N>&);
    template<class Type, std::size_t N>
    linalg::Vector<Type, N> gradient(const ForwardOnReverse<Type, N>&);
    template<class Type, std::size_t N, class ...Args>
    linalg::Vector<Type, N> gradient(const ForwardOnReverse<Type, N>&, Args&&...);
    template<class Type, class ...Args>
    linalg::Vector<Type, sizeof...(Args)> gradient(const ReverseOnReverse<Type>&, Args&&...);

    template<class Type, std::size_t N>
    linalg::Matrix<Type, N, N> hessian(const ForwardOnForward<Type, N>&);
    template<class Type, std::size_t N>
    linalg::Matrix<Type, N, N> hessian(const ReverseOnForward<Type, N>&, const std::array<ReverseOnForward<Type, N>, N>&);
    template<class Type, std::size_t N>
    linalg::Matrix<Type, N, N> hessian(const ForwardOnReverse<Type, N>&, const std::array<ForwardOnReverse<Type, N>, N>&);
    template<class Type, std::size_t N>
    linalg::Matrix<Type, N, N>
    hessian(const ReverseOnReverse<Type>&, const ReverseOnReverse<Type> (&x)[N]);
    template<class Type, std::size_t N>
    linalg::Matrix<Type, N, N>
    hessian(const ReverseOnReverse<Type>&, const std::array<ReverseOnReverse<Type>, N>&);
    template<class Type, class ...Args>
    linalg::Matrix<Type, 1 + sizeof...(Args), 1 + sizeof...(Args)>
    hessian(const ReverseOnReverse<Type>&, const ReverseOnReverse<Type>&, Args&&...);
}

template<class Type, std::size_t N>
linalg::Vector<Type, N> autodiff::gradient(const autodiff::Forward<Type, N>& fx)
{
    return fx.gradient();
}

template<class Type, class ...Args>
linalg::Vector<Type, sizeof...(Args)> autodiff::gradient(const autodiff::Reverse<Type>& fx, Args&& ...x)
{
    using SelfType = autodiff::Reverse<Type>;
    using SizeType = typename SelfType::SizeType;

    constexpr SizeType N = sizeof...(Args);
    static_assert(N > 0, "gradient must be computed with respect to at least 1 argument");
    static_assert((std::is_same_v<SelfType, std::decay_t<Args>> && ...), "all of the arguments should have the same type");
    
    linalg::Vector<Type, N> gradient_{{fx.partialDerivativeWRT(x)...}}; 
    return gradient_;
}

template<class Type, std::size_t N>
linalg::Vector<Type, N> autodiff::gradient(const ForwardOnForward<Type, N>& fx)
{
    return fx.value().gradient();
}

template<class Type, std::size_t N>
linalg::Vector<Type, N> autodiff::gradient(const ReverseOnForward<Type, N>& fx)
{
    return fx.value().gradient();
}

template<class Type, std::size_t N>
linalg::Vector<Type, N> autodiff::gradient(const ForwardOnReverse<Type, N>& fx)
{
    using VectorType = linalg::Vector<Type, N>;
    using SizeType = typename VectorType::SizeType; 

    VectorType gradient_;
    auto fx_gradient = fx.gradient();
    for(SizeType i = 0; i < N; ++i)
    {
        gradient_[i] = fx_gradient[i].value();
    }
    return gradient_;
}

template<class Type, std::size_t N, class ...Args>
linalg::Vector<Type, N> autodiff::gradient(const ForwardOnReverse<Type, N>& fx, Args&& ...x)
{
    using SelfType = ForwardOnReverse<Type, N>;

    static_assert(N == sizeof...(Args), "number of variables should be equal to N");
    static_assert((std::is_same_v<SelfType, std::decay_t<Args>> && ...), "all of the arguments should have the same type");

    return gradient(fx.value(), x.value()...);
}

template<class Type, class ...Args>
linalg::Vector<Type, sizeof...(Args)> autodiff::gradient(const ReverseOnReverse<Type>& fx, Args&& ...x)
{
    using SelfType = ReverseOnReverse<Type>;

    static_assert((std::is_same_v<SelfType, std::decay_t<Args>> && ...), "all of the arguments should have the same type");

    return gradient(fx.value(), x.value()...);
}

template<class Type, std::size_t N>
linalg::Matrix<Type, N, N> autodiff::hessian(const ForwardOnForward<Type, N>& fx)
{   // gradient of the derivative
    using MatrixType = linalg::Matrix<Type, N, N>;
    using SizeType = typename MatrixType::SizeType;

    MatrixType hessian_;
    linalg::ColumnProxy<MatrixType> column(hessian_);
    auto derivative = fx.gradient();
    for(SizeType j = 0; j < N; ++j)
    {
        column[j] = derivative[j].gradient();
    }
    return hessian_;
}

template<class Type, std::size_t N>
linalg::Matrix<Type, N, N> autodiff::hessian(const ReverseOnForward<Type, N>& fx, const std::array<ReverseOnForward<Type, N>, N>& x)
{
    using SizeType = typename ReverseOnForward<Type, N>::SizeType;
    using MatrixType = linalg::Matrix<Type, N, N>;

    MatrixType hessian_;
    linalg::ColumnProxy<MatrixType> column(hessian_);
    for(SizeType j = 0; j < N; ++j)
    {
        auto d_fx_d_xj = fx.partialDerivativeWRT(x[j]);
        column[j] = d_fx_d_xj.gradient();
    }
    return hessian_;
}

template<class Type, std::size_t N>
linalg::Matrix<Type, N, N> autodiff::hessian(const ForwardOnReverse<Type, N>& fx, const std::array<ForwardOnReverse<Type, N>, N>& x)
{
    using SizeType = typename ForwardOnReverse<Type, N>::SizeType;

    using MatrixType = linalg::Matrix<Type, N, N>;
    MatrixType hessian_;
    auto derivative = fx.gradient();
    for(SizeType j = 0; j < N; ++j)
    {
        for(SizeType i = 0; i < N; ++i)
        {
            hessian_(i, j) = derivative[j].partialDerivativeWRT(x[i].value());
        }
    }    
    return hessian_;
}

// template argument deduction fails with std::array
template<class Type, std::size_t N>
linalg::Matrix<Type, N, N>
autodiff::hessian(const ReverseOnReverse<Type>& fx, const ReverseOnReverse<Type> (&x)[N])
{
    return hessian(fx, std::to_array(x));
}

template<class Type, std::size_t N>
linalg::Matrix<Type, N, N>
autodiff::hessian(const ReverseOnReverse<Type>& fx, const std::array<ReverseOnReverse<Type>, N>& x)
{
    using SizeType = typename ReverseOnReverse<Type>::SizeType;

    linalg::Matrix<Type, N, N> hessian_;
    for(SizeType j = 0; j < N; ++j)
    {
        auto d_fx_d_xj = fx.partialDerivativeWRT(x[j]);
        for(SizeType i = 0; i < N; ++i)
        {
            hessian_(i, j) = d_fx_d_xj.partialDerivativeWRT(x[i].value());
        }
    }
    return hessian_;
}

template<class Type, class ...Args>
linalg::Matrix<Type, 1 + sizeof...(Args), 1 + sizeof...(Args)>
autodiff::hessian(const ReverseOnReverse<Type>& fx, const ReverseOnReverse<Type>& x0, Args&& ...xs)
{
    using VariableType = ReverseOnReverse<Type>;
    using SizeType = typename VariableType::SizeType;

    constexpr SizeType N = 1 + sizeof...(Args);
    static_assert(N > 0, "gradient must be computed with respect to at least 1 argument");
    static_assert((std::is_same_v<VariableType, std::decay_t<Args>> && ...), "all of the arguments should have the same type");

    std::array<VariableType, N> x{{x0, xs...}};
    return hessian(fx, x);
}
