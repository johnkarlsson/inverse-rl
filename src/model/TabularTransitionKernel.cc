#include "TabularTransitionKernel.h"

TabularTransitionKernel::TabularTransitionKernel(int _states, int _actions)
    : TransitionKernel(_states, _actions)
{
    kernel = std::vector<std::vector<std::vector<double>>>(
            states, std::vector<std::vector<double>>(
                actions, std::vector<double>(
                    states)));
}

TabularTransitionKernel::~TabularTransitionKernel()
{}

double TabularTransitionKernel::getTransitionProbability(const int s,
                                             const int a,
                                             const int s2) const
{
    return kernel[s][a][s2];
}

void TabularTransitionKernel::setTransitionProbability(const int s,
                                             const int a,
                                             const int s2,
                                             const double p)
{
    kernel[s][a][s2] = p;
}
