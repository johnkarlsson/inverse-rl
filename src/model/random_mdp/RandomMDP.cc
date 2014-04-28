#include "RandomMDP.h"
#include "RandomCMP.h"
#include "../DiscreteMDP.h"
#include "RandomTransitionKernel.h"

#include <vector>

RandomMDP::RandomMDP(const RandomCMP *cmp, double gamma)
    : DiscreteMDP(cmp, gamma)
{
    // Init random rewards.
    auto kernel = (RandomTransitionKernel*) cmp->kernel;
    // Works, but originally used for sampling transition probabilities.
    std::vector<double> rewards = kernel->sample_multinomial(); 
    for (int s = 0; s < cmp->states; ++s)
        setReward(s, rewards[s]);
}
