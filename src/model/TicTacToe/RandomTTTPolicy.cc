#include "RandomTTTPolicy.h"
#include "TicTacToeCMP.h"

#include <vector>
#include <random>

RandomTTTPolicy::RandomTTTPolicy(TicTacToeCMP const * _cmp, bool _playWin)
    : cmp(_cmp), playWin(_playWin)
{
    srand((unsigned)time(NULL));
}

int RandomTTTPolicy::action(TicTacToeCMP::State s)
{
    std::set<int> validActionsSet = cmp->kernel->getValidActions(s.getState());
    std::vector<int> validActions(validActionsSet.size());
    std::copy(validActionsSet.begin(), validActionsSet.end(),
              std::back_inserter(validActions));

    const int player = (validActions.size() % 2 == 0) ? 2 : 2;
    if (validActions.size() == 0)
        throw std::invalid_argument(
            "OptimalTTTPolicy.action() called with full board.");

    std::vector< std::vector<double> > features(validActions.size());
    std::vector<double> baseFeatures = cmp->features(s);
    for (int i = 0; i < validActions.size(); ++i)
    {
        int a = validActions[i];
        TicTacToeCMP::State sa(s);
        sa.move(a, 1);
        features[i] = cmp->features(sa);
    }

    // Win
    const int win_feature = (player == 1) ?
        TicTacToeCMP::FEATURE_TRIPLETS_X : TicTacToeCMP::FEATURE_TRIPLETS_O;
    if (playWin)
        for (int i = 0; i < validActions.size(); ++i)
            if (features[i][win_feature] > 0)
                return validActions[i];

    return validActions[rand()%validActions.size()];
}
