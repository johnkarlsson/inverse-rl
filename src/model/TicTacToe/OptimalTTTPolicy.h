#ifndef OPTIMALTTTPOLICY_H
#define OPTIMALTTTPOLICY_H

#include "TicTacToeCMP.h"

// Bare minimum class.
class OptimalTTTPolicy
{
    public:
        OptimalTTTPolicy(TicTacToeCMP const * _cmp);
        int action(TicTacToeCMP::State s);
    private:
        TicTacToeCMP const * cmp;
};

#endif
