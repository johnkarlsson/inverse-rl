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

RandomMDP::RandomMDP(const RandomCMP *cmp, double gamma)
    : DiscreteMDP(cmp, gamma), rewardWeights(cmp->nFeatures())
{
    // Init random rewards.
    auto kernel = (RandomTransitionKernel*) cmp->kernel;
    // Works, but originally used for sampling transition probabilities.
    auto weights = kernel->sample_multinomial();

    int s;
    for (s = 0; s < cmp->states; ++s)
        setReward(s, weights[s]);
    setReward(s, 0);

    // for (s = 0; s < cmp->states; ++s)
    // {
    //     cout << "rewardWeights[" << s << "]==\t" << rewardWeights[s] << endl;
    //     cout << "weights[" << s << "]\t==\t" << rewardWeights[s] << endl;
    //     cout << ", getReward(" << s << ")\t==\t" << getReward(s) << endl;
    // }

    /*
    std::vector<double> rewards = kernel->sample_multinomial(); 
    for (int s = 0; s < cmp->states; ++s)
        setReward(s, rewards[s]);
    */
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

double RandomMDP::getReward(int s) const
{
    auto phi = cmp->features(s);
    return inner_product(phi.begin(), phi.end(), rewardWeights.begin(), 0.0);
}
