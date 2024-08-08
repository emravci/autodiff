#pragma once

#include <vector>
#include <memory>
#include <map>
#include <cmath>
#include <cassert>

namespace autodiff 
{ 
    template<class Type> class Reverse; 
    template<class Type> class PerfectBinaryTree;
    template<class Type> constexpr Type one = 1;
    template<class Type> Reverse<Type> sin(const Reverse<Type>&);
    template<class Type> Reverse<Type> cos(const Reverse<Type>&);
    template<class Type> Reverse<Type> tan(const Reverse<Type>&);
    template<class Type> Reverse<Type> exp(const Reverse<Type>&);
    template<class Type> Reverse<Type> log(const Reverse<Type>&);
    template<class Type> Reverse<Type> pow(const Reverse<Type>&, const Reverse<Type>&);
}

template<class Type>
class autodiff::PerfectBinaryTree
{
    public:
    using SelfType = PerfectBinaryTree<Type>;
    using SizeType = std::size_t;
    PerfectBinaryTree(SizeType levelCount) : nodes_(levelCountToNodeCount(levelCount)) {}
    PerfectBinaryTree(Type, const SelfType&);
    PerfectBinaryTree(Type, const SelfType&, const SelfType&);    
    SizeType nodeCount() const { return nodes_.size(); }
    SizeType levelCount() const { return std::log2(nodeCount() + 1); }
    Type& getParentOf(SizeType);
    const Type& getParentOf(SizeType) const;
    Type& operator[](SizeType i) { return nodes_[i]; }
    const Type& operator[](SizeType i) const { return nodes_[i]; }
    Type& operator()(SizeType level, SizeType i) { return nodes_[nodeCountBeforeLevel(level) + i]; }
    const Type& operator()(SizeType level, SizeType i) const { return nodes_[nodeCountBeforeLevel(level) + i]; }
    static bool isLeftChild(SizeType i) { return i % 2 == 1; }
    static bool isRightChild(SizeType i) { return i % 2 == 0; }
    static SizeType getParentIndexOf(SizeType child) { return (child - 1) / 2; }
    static SizeType nodeCountAtLevel(SizeType i) { return one<SizeType> << i; }
    static SizeType nodeCountBeforeLevel(SizeType i) { return (one<SizeType> << i) - one<SizeType>; }
    static SizeType levelCountToNodeCount(SizeType levelCount) { return (one<SizeType> << levelCount) - one<SizeType>; }
    private:
    void copyTreeAsBranchOfRoot(const SelfType&, bool);
    std::vector<Type> nodes_;
};

template<class Type>
autodiff::PerfectBinaryTree<Type>::PerfectBinaryTree(Type root, const SelfType& leftBranch)
: nodes_(levelCountToNodeCount(1 + leftBranch.levelCount()))
{
    nodes_[0] = std::move(root);
    copyTreeAsBranchOfRoot(leftBranch, false);
}

template<class Type>
autodiff::PerfectBinaryTree<Type>::PerfectBinaryTree(Type root, const SelfType& leftBranch, const SelfType& rightBranch)
: nodes_(levelCountToNodeCount(1 + std::max(leftBranch.levelCount(), rightBranch.levelCount())))
{
    nodes_[0] = std::move(root);
    copyTreeAsBranchOfRoot(leftBranch, false);
    copyTreeAsBranchOfRoot(rightBranch, true);
}

template<class Type>
Type& autodiff::PerfectBinaryTree<Type>::getParentOf(SizeType child)
{
    assert(child != 0 && "root has no parent");
    return nodes_[getParentIndexOf(child)];
}

template<class Type>
const Type& autodiff::PerfectBinaryTree<Type>::getParentOf(SizeType child) const
{
    assert(child != 0 && "root has no parent");
    return nodes_[getParentIndexOf(child)];
}

template<class Type>
void autodiff::PerfectBinaryTree<Type>::copyTreeAsBranchOfRoot(const SelfType& branch, bool isRightBranch)
{
    SelfType &self = *this;
    for(SizeType i = 0, height = branch.levelCount(); i < height; ++i)
    {
        for(SizeType j = 0, nodeCountAtLevel_i = nodeCountAtLevel(i); j < nodeCountAtLevel_i; ++j)
        {
            self(i + 1, j + nodeCountAtLevel_i * isRightBranch) = branch(i, j);
        }
    }
}

template<class Type>
class autodiff::Reverse
{
    public:
    struct Node;
    struct Partial;
    using SelfType = Reverse<Type>;
    using SizeType = std::size_t;
    using SharedNodeType = std::shared_ptr<Node>;
    using TreeType = PerfectBinaryTree<SharedNodeType>;
    using MapType = std::multimap<Node*, SizeType>;
    Reverse();
    Reverse(Type value);
    Reverse(Node node, const SelfType& that);
    Reverse(Node node, const SelfType& lhs, const SelfType& rhs);
    Type value() const { return root()->value_; }
    Type partialDerivativeWRT(const SelfType&) const;
    private:
    SharedNodeType& root() { return tree_[0]; }
    const SharedNodeType& root() const { return tree_[0]; }
    void fillEmptyMapWithTree();
    Type multiplyPartialDerivativesAlongChain(SizeType) const;
    const Type& getPartialDerivativeOfNodeWRT(SizeType) const;
    TreeType tree_;
    MapType map_;
};

template<class Type>
struct autodiff::Reverse<Type>::Partial
{
    Type leftOperand_{}, rightOperand_{};
};

template<class Type>
struct autodiff::Reverse<Type>::Node 
{
    Node(Type value = {}, Type dLeft = {}, Type dRight = {}) 
    : value_{value}, partialDerivativeWRT_{dLeft, dRight} {}
    Type value_; 
    Partial partialDerivativeWRT_; 
};

template<class Type>
autodiff::Reverse<Type>::Reverse() : tree_(1) 
{
    root() = std::make_shared<Node>();
}

template<class Type>
autodiff::Reverse<Type>::Reverse(Type value) : tree_(1)
{
    root() = std::make_shared<Node>(value);
    fillEmptyMapWithTree();
}

template<class Type>
autodiff::Reverse<Type>::Reverse(Node node, const SelfType& that) 
: tree_{std::make_shared<Node>(std::move(node)), that.tree_}
{
    fillEmptyMapWithTree();
}

template<class Type>
autodiff::Reverse<Type>::Reverse(Node node, const SelfType& lhs, const SelfType& rhs) 
: tree_{std::make_shared<Node>(std::move(node)), lhs.tree_, rhs.tree_}
{
    fillEmptyMapWithTree();
}

template<class Type>
void autodiff::Reverse<Type>::fillEmptyMapWithTree()
{
    assert(map_.empty() && "this function can only be called in ctor when map is empty");
    for(SizeType i = 0; i < tree_.nodeCount(); ++i)
    {
        if(tree_[i].get() != nullptr) { map_.insert({tree_[i].get(), i}); }        
    }
}

template<class Type>
Type autodiff::Reverse<Type>::multiplyPartialDerivativesAlongChain(SizeType variable) const
{
    constexpr SizeType root = 0;
    Type chainRule(1);
    for(SizeType child = variable; child != root; child = TreeType::getParentIndexOf(child))
    {
        chainRule = getPartialDerivativeOfNodeWRT(child) * chainRule;
    }
    return chainRule;
}

template<class Type>
const Type& autodiff::Reverse<Type>::getPartialDerivativeOfNodeWRT(SizeType child) const
{
    if(TreeType::isLeftChild(child)) { return tree_.getParentOf(child)->partialDerivativeWRT_.leftOperand_; }
    else { return tree_.getParentOf(child)->partialDerivativeWRT_.rightOperand_; }
}

template<class Type>
Type autodiff::Reverse<Type>::partialDerivativeWRT(const SelfType& g) const 
{
    auto gRoot = g.root().get();
    Type partialDerivativeWRT_g(0);
    for(auto [it, end] = map_.equal_range(gRoot); it != end; ++it)
    {
        auto indexOfVariableOfInterest = it->second;
        partialDerivativeWRT_g = partialDerivativeWRT_g + multiplyPartialDerivativesAlongChain(indexOfVariableOfInterest);
    }
    return partialDerivativeWRT_g;
}

template<class Type>
autodiff::Reverse<Type> operator+(const autodiff::Reverse<Type>& lhs, const autodiff::Reverse<Type>& rhs)
{
    using OperandType = autodiff::Reverse<Type>;
    using NodeType = typename OperandType::Node;
    return OperandType(NodeType(lhs.value() + rhs.value(), Type(1), Type(1)), lhs, rhs);
}

template<class Type>
autodiff::Reverse<Type> operator-(const autodiff::Reverse<Type>& that)
{
    return that * autodiff::Reverse<Type>(-1);
}

template<class Type>
autodiff::Reverse<Type> operator-(const autodiff::Reverse<Type>& lhs, const autodiff::Reverse<Type>& rhs)
{
    using OperandType = autodiff::Reverse<Type>;
    using NodeType = typename OperandType::Node;
    return OperandType(NodeType(lhs.value() - rhs.value(), Type(1), Type(-1)), lhs, rhs);
}

template<class Type>
autodiff::Reverse<Type> operator*(const autodiff::Reverse<Type>& lhs, const autodiff::Reverse<Type>& rhs)
{
    using OperandType = autodiff::Reverse<Type>;
    using NodeType = typename OperandType::Node;
    return OperandType(NodeType(lhs.value() * rhs.value(), rhs.value(), lhs.value()), lhs, rhs);
}

template<class Type>
autodiff::Reverse<Type> operator/(const autodiff::Reverse<Type>& lhs, const autodiff::Reverse<Type>& rhs)
{
    using OperandType = autodiff::Reverse<Type>;
    using NodeType = typename OperandType::Node;
    return OperandType(NodeType(lhs.value() / rhs.value(), Type(1) / rhs.value(), -lhs.value() / rhs.value() / rhs.value()), lhs, rhs);
}

template<class Type>
autodiff::Reverse<Type> autodiff::sin(const autodiff::Reverse<Type>& that)
{
    using OperandType = autodiff::Reverse<Type>;
    using NodeType = typename OperandType::Node;
    using std::sin, std::cos;
    return OperandType(NodeType(sin(that.value()), cos(that.value())), that);
}

template<class Type>
autodiff::Reverse<Type> autodiff::cos(const autodiff::Reverse<Type>& that)
{
    using OperandType = autodiff::Reverse<Type>;
    using NodeType = typename OperandType::Node;
    using std::sin, std::cos;
    return OperandType(NodeType(cos(that.value()), -sin(that.value())), that);
}

template<class Type>
autodiff::Reverse<Type> autodiff::tan(const autodiff::Reverse<Type>& that)
{
    using OperandType = autodiff::Reverse<Type>;
    using NodeType = typename OperandType::Node;
    using std::tan, std::sin, std::cos;
    return OperandType(NodeType(tan(that.value()), Type(1) / cos(that.value() / cos(that.value()))), that);
}

template<class Type>
autodiff::Reverse<Type> autodiff::exp(const Reverse<Type>& that)
{
    using OperandType = autodiff::Reverse<Type>;
    using NodeType = typename OperandType::Node;
    using std::exp;
    return OperandType(NodeType(exp(that.value()), exp(that.value())), that);
}

template<class Type>
autodiff::Reverse<Type> autodiff::log(const Reverse<Type>& that)
{
    using OperandType = autodiff::Reverse<Type>;
    using NodeType = typename OperandType::Node;
    using std::log;
    return OperandType(NodeType(log(that.value()), Type(1) / that.value()), that);
}

template<class Type>
autodiff::Reverse<Type> autodiff::pow(const Reverse<Type>& x, const Reverse<Type>& y)
{
    using OperandType = autodiff::Reverse<Type>;
    using NodeType = typename OperandType::Node;
    using std::pow, std::log;
    return OperandType(NodeType(
        pow(x.value(), y.value()), 
        pow(x.value(), y.value() - Type(1)), pow(x.value(), y.value()) * log(x.value())), x, y);
}

#if 0
template<class Type>
std::ostream& operator<<(std::ostream& out, const autodiff::Reverse<Type>& that)
{
	using SizeType = typename autodiff::Reverse<Type>::SizeType;
    auto tree = that.tree();
    out << "[ ";
    for(SizeType i = 0; i < tree.nodeCount(); ++i) 
    {
        if(tree[i].get() == nullptr) { continue; }
        out << i << "x V";
        out << tree[i]->value_ << " L"; 
        out << tree[i]->partialDerivativeWRT_.leftOperand_ << " R";
        out << tree[i]->partialDerivativeWRT_.rightOperand_ << " ";
    }
    out << "]";
	return out;
}
#endif
