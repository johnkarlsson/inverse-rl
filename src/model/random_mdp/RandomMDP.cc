#include "RandomMDP.h"
#include "RandomCMP.h"
#include "../DiscreteMDP.h"
#include "RandomTransitionKernel.h"

#include <vector>
#include <numeric>

#include <iostream>
using namespace std;

using std::vector;
using std::inner_product;

RandomMDP::RandomMDP(const DiscreteCMP *cmp, double gamma)
    : FeatureMDP(cmp, gamma)//, rewardWeights(cmp->nFeatures())
{
    // Init random rewards.
    auto kernel = (RandomTransitionKernel*) cmp->kernel;
    // Works, but originally used for sampling transition probabilities.
    auto weights = kernel->sample_multinomial();

    setRewardWeights(weights);
}

/*
double RandomMDP::getReward(int s) const
{
    auto phi = cmp->features(s);
    return inner_product(phi.begin(), phi.end(), rewardWeights.begin(), 0.0);
}

vector<double> RandomMDP::getRewardWeights()
{
    return rewardWeights;
}

void RandomMDP::setRewardWeights(vector<double> weights)
{
    rewardWeights = weights;
}

void RandomMDP::setReward(int s, double r)
{
    rewardWeights[s] = r;
}
*/
