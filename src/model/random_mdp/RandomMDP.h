#ifndef RANDOMMDP_H
#define RANDOMMDP_H

#include "RandomCMP.h"
#include "../DiscreteMDP.h"

class RandomMDP
    : public DiscreteMDP
{
    public:
        RandomMDP(const RandomCMP * cmp, double gamma);

    private:
};

#endif
