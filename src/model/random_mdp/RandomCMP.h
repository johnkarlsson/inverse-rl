#ifndef RANDOMCMP_H
#define RANDOMCMP_H

#include "../DiscreteCMP.h"
#include "RandomTransitionKernel.h"

class RandomCMP : public DiscreteCMP
{
    public:
        RandomCMP(RandomTransitionKernel const * kernel);
};

#endif
