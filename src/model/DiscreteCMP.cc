#include "DiscreteCMP.h"

DiscreteCMP::DiscreteCMP(int _states, int _actions)
    : states(_states), actions(_actions)
{
    kernel = std::vector<std::vector<std::vector<double>>>(
            states, std::vector<std::vector<double>>(
                actions, std::vector<double>(
                    states)));
}

double DiscreteCMP::getTransitionProbability(const int s,
                                             const int a,
                                             const int s2) const
{
    return kernel[s][a][s2];
}

void DiscreteCMP::setTransitionProbability(const int s,
                                             const int a,
                                             const int s2,
                                             const double p)
{
    kernel[s][a][s2] = p;
}
