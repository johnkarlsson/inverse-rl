#include "OptimalTTTPolicy.h"
#include "TicTacToeCMP.h"

#include <vector>
#include <iostream>

OptimalTTTPolicy::OptimalTTTPolicy(TicTacToeCMP const * _cmp)
    : cmp(_cmp)
{}

int OptimalTTTPolicy::action(TicTacToeCMP::State s)
{
    std::vector<int> validActions = cmp->kernel->getValidActions(s.getState());
    std::vector< std::vector<double> > features(validActions.size());
    int ai = 0;
    for (int a : validActions)
    {
        TicTacToeCMP::State sa(s);
        sa.move(a, 1);
        features[ai++] = cmp->features(sa);
    }

    ai = 0;
    for (int a : validActions)
    {
        bool win = features[ai++][TicTacToeCMP::FEATURE_TRIPLETS_X] > 0;

        if (win)
            return a;
    }

    return validActions[0];
}
