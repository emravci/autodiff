#pragma once

#include <array> 

#include "typedefs.hpp"
#include "Forward.hpp"
#include "Reverse.hpp"

namespace autodiff
{    
    template<class Type> struct Make;
    
    template<class Type>
    constexpr bool IsReverse = false;

    template<class Type>
    constexpr bool IsReverse<Reverse<Type>> = true;

    template<class Type>
    constexpr bool IsForward = false;

    template<class Type, std::size_t N>
    constexpr bool IsForward<Forward<Type, N>> = true;

    template<class Type>
    concept AutoDifferentiable = IsForward<Type> || IsReverse<Type>;

    template<class Ty> struct ValueType
    {
        using Type = Ty;
    };
    
    template<class Ty>
    struct ValueType<Reverse<Ty>>
    {
        using Type = typename ValueType<Ty>::Type;
    };

    template<class Ty, std::size_t N>
    struct ValueType<Forward<Ty, N>>
    {
        using Type = typename ValueType<Ty>::Type;
    };

    template<class Type, std::size_t N>
    struct Make<Forward<Type, N>>
    {
        using SizeType = typename Forward<Type, N>::SizeType;
        static Forward<Type, N> variable(Type value, SizeType i)
        {
            return Forward<Type, N>(value, i);
        }
        static Forward<Type, N> constant(Type value)
        {
            return Forward<Type, N>(value);
        }
    };

    template<AutoDifferentiable Type, std::size_t N>
    struct Make<Forward<Type, N>>
    {
        using ValueType = typename ValueType<Type>::Type;
        using SizeType = typename Forward<Type, N>::SizeType;
        static Forward<Type, N> variable(ValueType value, SizeType i)
        {
            return Forward<Type, N>(Make<Type>::variable(value, i), i);
        }
        static Forward<Type, N> constant(ValueType value)
        {
            return Forward<Type, N>(Make<Type>::constant(value));
        }
    };
    
    template<class Type>
    struct Make<Reverse<Type>>
    {
        using SizeType = typename Reverse<Type>::SizeType;
        static Reverse<Type> variable(Type value, SizeType i)
        {
            return Reverse<Type>(value);
        }
    };

    template<AutoDifferentiable Type>
    struct Make<Reverse<Type>>
    {
        using ValueType = typename ValueType<Type>::Type;
        using SizeType = typename Reverse<Type>::SizeType;
        static Reverse<Type> variable(ValueType value, SizeType i)
        {
            return Reverse<Type>(Make<Type>::variable(value, i));
        }
    };
}

