#include "TransitionKernel.h"
#include <string>
#include <stdexcept>

TransitionKernel::TransitionKernel(int _states, int _actions)
    : states(_states), actions(_actions)
{}

/*
std::set<int> TransitionKernel::getValidActions(const int s) const
{
    throw std::runtime_error("Undefined function TransitionKernel::getValidActions");
}

std::vector< std::pair<state, probability> >
    TransitionKernel::getTransitionProbabilities(const int s, const int a) const
{
    throw std::runtime_error("Undefined function TransitionKernel::getTransitionProbabilities");
}
*/
