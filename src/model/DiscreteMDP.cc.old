#include "DiscreteMDP.h"

DiscreteMDP::DiscreteMDP(const DiscreteCMP *_cmp, double _gamma)
    : cmp(_cmp), gamma(_gamma)
{
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
