#include "RandomCMP.h"
#include "RandomTransitionKernel.h"
#include "../DiscreteCMP.h"

// #include <cmath>
// #include <vector>
// #include <iostream>
// #include <sstream>

RandomCMP::RandomCMP(RandomTransitionKernel const * kernel)
    : DiscreteCMP(kernel)
{}
