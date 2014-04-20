#include "DiscreteCMP.h"
#include "TransitionKernel.h"

DiscreteCMP::DiscreteCMP(TransitionKernel const *_kernel)
    : kernel(_kernel), states(_kernel->states), actions(_kernel->actions)
{}

DiscreteCMP::~DiscreteCMP()
{}
