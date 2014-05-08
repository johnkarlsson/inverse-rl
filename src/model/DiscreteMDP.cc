#include "DiscreteMDP.h"

DiscreteMDP::DiscreteMDP(const DiscreteCMP *_cmp, double _gamma,
                         bool initRewardVector)
    : cmp(_cmp), gamma(_gamma)
{
    if (initRewardVector)
        rewards = std::vector<double>(cmp->states);
}

double DiscreteMDP::getReward(int s) const
{
    return rewards[s];
}

void DiscreteMDP::setReward(int s, double r)
{
    rewards[s] = r;
}
