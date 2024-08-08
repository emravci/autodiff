#pragma once

#include <cassert>
#include <cstddef>

namespace linalg
{
    template<class Type, std::size_t M, std::size_t N> class Matrix;
    template<class Type, std::size_t N> class Vector;
    template<class Type> class ColumnProxy;
}

template<class Type, std::size_t M, std::size_t N>
class linalg::ColumnProxy<linalg::Matrix<Type, M, N>>
{
    public:
    using MatrixType = Matrix<Type, M, N>;
    using SelfType = ColumnProxy<MatrixType>;
    using SizeType = typename MatrixType::SizeType;
    ColumnProxy(MatrixType& matrix) : matrix_{matrix} {}
    SelfType& operator[](SizeType j) 
    {
        assert(j < N && "cannot be equal to or greater than number of columns");
        column_j = j;
        return *this;
    }
    SelfType& operator=(const Vector<Type, M>& vector)
    {
        for(SizeType i = 0; i < M; ++i)
        {
            matrix_(i, column_j) = vector[i];
        }
        return *this;
    }
    private:
    MatrixType &matrix_;
    SizeType column_j;
};
