#include "DiscreteCMP.h"
#include "TransitionKernel.h"

#include <vector>
#include <stdexcept>

#include <iostream>
using std::cout;

DiscreteCMP::DiscreteCMP(TransitionKernel const *_kernel)
    : kernel(_kernel), states(_kernel->states), actions(_kernel->actions)
{}

const bool DEBUG_SAFEATURES = false;

int DiscreteCMP::nFeatures() const
{
    if (DEBUG_SAFEATURES)
        return states*actions;
    return states;
    return states + 1;
}

DiscreteCMP::~DiscreteCMP()
{}

std::vector<double> DiscreteCMP::features(int s) const
{
    if (DEBUG_SAFEATURES)
        throw std::runtime_error("DiscreteCMP::features(int) is undefined");

    std::vector<double> output(nFeatures(), 0);
    output[s] = 1;
    // output[nFeatures() - 1] = 1;
    return output;
}

std::vector<double> DiscreteCMP::features(int s, int a) const
{
    /*
    std::vector<double> phi(nFeatures(), 0);
    phi[s] = 1;
    return phi;
    phi[s*actions + a] = 1; // |S| x |A| matrix
    return phi;
    */

    // s * a
    // (s-1)(a-1) = sa - s - a 



    std::vector<double> phiAvg(nFeatures(), 0);

    // auto actions = kernel->getValidActions(s);
    // for (auto a : actions)
    // {
        auto transitionProbabilities = kernel->getTransitionProbabilities(s, a);
        // if (s == 2 && a == 0)
        // {
        //     for (std::pair<int,double> sp : transitionProbabilities)
        //     {
        //         cout << "!" << sp.first << "!" << sp.second << std::endl;
        //     }
        // }
        for (std::pair<int,double> sp : transitionProbabilities)
        {
            int s2 = sp.first;
            double p = sp.second;

            auto phi2 = features(s2);
            for (int i = 0; i < nFeatures(); ++i)
                phiAvg[i] += p * phi2[i];
        }
    // }

    return phiAvg;
}
