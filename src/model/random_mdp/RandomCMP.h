#ifndef RANDOMCMP_H
#define RANDOMCMP_H

#include "../DiscreteCMP.h"
#include "RandomTransitionKernel.h"
// #include <vector>
// #include <cmath>
// #include <stdexcept>

class RandomCMP : public DiscreteCMP
{
    public:
        RandomCMP(RandomTransitionKernel const * kernel);
};

#endif
