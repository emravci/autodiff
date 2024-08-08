#pragma once

namespace autodiff
{
    template<class Type, std::size_t N> class Forward;
    template<class Type> class Reverse; 
    
    template<class Type, std::size_t N>
    using ForwardOnForward = Forward<Forward<Type, N>, N>;
    template<class Type, std::size_t N>
    using ForwardOnReverse = Forward<Reverse<Type>, N>;
    template<class Type, std::size_t N>
    using ReverseOnForward = Reverse<Forward<Type, N>>;
    template<class Type>
    using ReverseOnReverse = Reverse<Reverse<Type>>;
}
