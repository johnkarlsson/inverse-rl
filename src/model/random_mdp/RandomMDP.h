#ifndef RANDOMMDP_H
#define RANDOMMDP_H

#include "RandomCMP.h"
#include "../DiscreteMDP.h"

#include <vector>

class RandomMDP
    : public FeatureMDP
{
    public:
        RandomMDP(const DiscreteCMP * cmp, double gamma);
};

#endif
