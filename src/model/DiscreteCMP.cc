#include "DiscreteCMP.h"
#include "TransitionKernel.h"

#include <vector>
#include <stdexcept>

#include <iostream>
using std::cout;

DiscreteCMP::DiscreteCMP(TransitionKernel const *_kernel)
    : states(_kernel->states), actions(_kernel->actions), kernel(_kernel)
{}

int DiscreteCMP::nFeatures() const
{
    return states;
}

DiscreteCMP::~DiscreteCMP()
{}

std::vector<double> DiscreteCMP::features(int s) const
{
    std::vector<double> output(nFeatures(), 0);
    output[s] = 1;

    return output;
}

std::vector<double> DiscreteCMP::features(int s, int a) const
{
    std::vector<double> phiAvg(nFeatures(), 0);

    auto transitionProbabilities = kernel->getTransitionProbabilities(s, a);
    for (std::pair<int,double> sp : transitionProbabilities)
    {
        int s2 = sp.first;
        double p = sp.second;

        auto phi2 = features(s2);
        for (int i = 0; i < nFeatures(); ++i)
            phiAvg[i] += p * phi2[i];
    }

    return phiAvg;
}
