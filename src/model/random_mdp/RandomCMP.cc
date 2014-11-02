#include "RandomCMP.h"
#include "RandomTransitionKernel.h"
#include "../DiscreteCMP.h"

RandomCMP::RandomCMP(RandomTransitionKernel const * kernel)
    : DiscreteCMP(kernel)
{}
