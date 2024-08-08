#pragma once

#include <array> 

#include "typedefs.hpp"
#include "Forward.hpp"
#include "Reverse.hpp"

namespace autodiff
{    
    template<class Type> struct Make;

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

    template<class Type, std::size_t N>
    struct Make<ForwardOnForward<Type, N>>
    {
        using SizeType = typename ForwardOnForward<Type, N>::SizeType;
        static ForwardOnForward<Type, N> variable(Type value, SizeType i)
        {
            return ForwardOnForward<Type, N>(Forward<Type, N>(value, i), i);
        }
        static ForwardOnForward<Type, N> constant(Type value)
        {
            return ForwardOnForward<Type, N>(Forward<Type, N>(value));
        }
    };
    
    template<class Type>
    struct Make<Reverse<Type>>
    {
        static Reverse<Type> variable(Type value)
        {
            return Reverse<Type>(value);
        }
    };   

    template<class Type, std::size_t N>
    struct Make<ForwardOnReverse<Type, N>>
    {
        using MadeType = ForwardOnReverse<Type, N>;
        using SizeType = typename MadeType::SizeType;
        static MadeType variable(Type value, SizeType i)
        {
            return MadeType(Reverse<Type>(value), i);
        }
        static std::array<MadeType, N> variables(const Type (&values)[N])
        {
            std::array<MadeType, N> variables_;
            for(SizeType i = 0; i < N; ++i)
            {
                variables_[i] = variable(values[i], i);
            }
            return variables_;
        }
    };

    template<class Type, std::size_t N>
    struct Make<ReverseOnForward<Type, N>>
    {
        using MadeType = ReverseOnForward<Type, N>;
        using SizeType = typename MadeType::SizeType;
        static MadeType variable(Type value, SizeType i)
        {
            return MadeType(Forward<Type, N>(value, i));
        }
        static std::array<MadeType, N> variables(const Type (&values)[N])
        {
            std::array<MadeType, N> variables_;
            for(SizeType i = 0; i < N; ++i)
            {
                variables_[i] = variable(values[i], i);
            }
            return variables_;
        }
    };

    template<class Type>
    struct Make<ReverseOnReverse<Type>>
    {
        static ReverseOnReverse<Type> variable(Type value)
        {
            return ReverseOnReverse<Type>(Reverse<Type>(value));
        }
    };
}

