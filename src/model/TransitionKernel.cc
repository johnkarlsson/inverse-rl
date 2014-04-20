#include "TransitionKernel.h"

TransitionKernel::TransitionKernel(int _states, int _actions)
    : states(_states), actions(_actions)
{}

std::vector<int> TransitionKernel::getValidActions(const int s) const
{
    return std::vector<int>(0);
}
