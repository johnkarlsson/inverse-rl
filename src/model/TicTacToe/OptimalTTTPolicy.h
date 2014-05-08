#ifndef OPTIMALTTTPOLICY_H
#define OPTIMALTTTPOLICY_H

#include "TicTacToeCMP.h"
#include "../Policy.h"

// Bare minimum class.
class OptimalTTTPolicy
    : public Policy
{
    public:
        OptimalTTTPolicy(TicTacToeCMP const * _cmp);
        int action(TicTacToeCMP::State const & s);
        std::vector<std::pair<int,double>> probabilities(int s);
    private:
        TicTacToeCMP const * cmp;
};

#endif
