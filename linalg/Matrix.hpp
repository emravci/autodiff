#pragma once

#include <array>

namespace linalg
{
    template<class Type, std::size_t M, std::size_t N> class Matrix;
}

template<class Type, std::size_t M, std::size_t N>
class linalg::Matrix
{
    public:
    using SizeType = std::size_t;
    Type& operator()(SizeType i, SizeType j) { return elements_[i * N + j]; }
    const Type& operator()(SizeType i, SizeType j) const { return elements_[i * N + j]; }
    private:
    std::array<Type, M * N> elements_;
};

template<class Type, std::size_t M, std::size_t N>
std::ostream& operator<<(std::ostream& out, const linalg::Matrix<Type, M, N>& that)
{
	using SizeType = typename linalg::Matrix<Type, M, N>::SizeType;
    out << "[";
    for(SizeType i = 0; i < M; ++i) 
    {
        out << "[ ";
        for(SizeType j = 0; j < N; ++j)
        {
            out << that(i, j) << " ";
        }
        out << "]";
    }        
    out << "]";
	return out;
}
