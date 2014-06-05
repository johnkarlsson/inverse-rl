#include "TicTacToeMDP.h"
#include "TicTacToeCMP.h"

#include <vector>

using std::vector;

TicTacToeMDP::TicTacToeMDP(const DiscreteCMP * cmp, bool _trueRewards)
    : FeatureMDP(cmp, 1.0), trueRewards(_trueRewards)
{}

double TicTacToeMDP::getReward(int s) const
{
    if (trueRewards)
    {
        const static double rewards[] = {0, 1, -1, 0}; // {!Terminal, X, O, Tie}
        int win = ((TicTacToeCMP*) cmp)->winner(
                TicTacToeCMP::State(((TicTacToeCMP*) cmp)->size, s));
        return rewards[win];
    }
    else
        return FeatureMDP::getReward(s);
}
