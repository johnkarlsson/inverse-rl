#ifndef GRIDTRANSITIONKERNEL_H
#define GRIDTRANSITIONKERNEL_H

#include "../TabularTransitionKernel.h"
#include <vector>

class GridTransitionKernel
    : public TabularTransitionKernel
{
    public:
        GridTransitionKernel(int size);
};

#endif
