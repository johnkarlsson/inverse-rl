#include "TicTacToeMDP.h"
#include "TicTacToeCMP.h"

#include <vector>
#include <numeric>

using std::vector;
using std::inner_product;

TicTacToeMDP::TicTacToeMDP(const DiscreteCMP *cmp, bool _trueRewards)
    : DiscreteMDP(cmp, 1.0, false), trueRewards(_trueRewards)
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
    {
        auto phi = cmp->features(s);
        return inner_product(phi.begin(), phi.end(), rewardWeights.begin(), 0.0);
    }
}

void TicTacToeMDP::setRewardWeights(vector<double> weights)
{
    rewardWeights = weights;
}

void TicTacToeMDP::setReward(int s, double r)
{
    throw std::runtime_error("Undefined function TicTacToeMDP::setReward");
    // rewardWeights[s] = r;
}


