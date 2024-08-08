#pragma once

namespace utility
{
    template<class Type> class Add;
    template<class Type> class Subtract;
    template<class Type> class Multiply;
    template<class Type> class Negate;
}

template<class Type>
struct utility::Add
{
    Type operator()(const Type& lhs, const Type& rhs) const
    {
        return lhs + rhs;
    }
};

template<class Type>
struct utility::Subtract
{
    Type operator()(const Type& lhs, const Type& rhs) const
    {
        return lhs - rhs;
    }
};

template<class Type>
struct utility::Multiply
{
    Type operator()(const Type& lhs, const Type& rhs) const
    {
        return lhs * rhs;
    }
};

template<class Type>
struct utility::Negate
{
    Type operator()(const Type& that) const
    {
        return -that;
    }
};
