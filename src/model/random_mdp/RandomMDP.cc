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
