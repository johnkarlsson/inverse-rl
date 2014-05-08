#include "OptimalTTTPolicy.h"
#include "TicTacToeCMP.h"

#include <vector>
#include <set>

OptimalTTTPolicy::OptimalTTTPolicy(TicTacToeCMP const * _cmp)
    : cmp(_cmp)
{}

std::vector<std::pair<int,double>> OptimalTTTPolicy::probabilities(int s)
{
    TicTacToeCMP::State state(cmp->size);
    const int player = (cmp->kernel->getValidActions(s).size() % 2 == 0) ? 2 : 1;
    if (player == 2)
        state.setState(s, true); // Hack: Invert board since action() is for X
    else
        state.setState(s);

    return {{action(state), 1}};
}

// NOTE: Only supports player X
int OptimalTTTPolicy::action(TicTacToeCMP::State const & s)
{
    std::set<int> validActionsSet = cmp->kernel->getValidActions(s.getState());
    std::vector<int> validActions;
    std::copy(validActionsSet.begin(), validActionsSet.end(),
              std::back_inserter(validActions));

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
    for (int i = 0; i < validActions.size(); ++i)
        if (features[i][TicTacToeCMP::FEATURE_TRIPLETS_X] > 0)
            return validActions[i];

    // Block
    for (int i = 0; i < validActions.size(); ++i)
        if (features[i][TicTacToeCMP::FEATURE_DOUBLETS_O]
            < baseFeatures[TicTacToeCMP::FEATURE_DOUBLETS_O])
            return validActions[i];

    // Fork
    for (int i = 0; i < validActions.size(); ++i)
        if (features[i][TicTacToeCMP::FEATURE_FORKS_X] > 0)
            return validActions[i];

    // Opponent features
    std::vector< std::vector<double> > featuresOpponent(validActions.size());
    for (int i = 0; i < validActions.size(); ++i)
    {
        int a = validActions[i];
        TicTacToeCMP::State sa(s);
        sa.move(a, 2);
        featuresOpponent[i] = cmp->features(sa);
    }

    // Block fork
    for (int i = 0; i < validActions.size(); ++i)
        if (featuresOpponent[i][TicTacToeCMP::FEATURE_FORKS_O] > 0) // >0 works
            return validActions[i];

    // Center
    for (int i = 0; i < validActions.size(); ++i)
        if (validActions[i] == s.size*s.size / 2)
            return validActions[i];

    // Opposite corner occupied by opponent
    int c = s.size - 1;
    int corners[4][2][2] = { { {0, c}, {c, 0} },
                             { {c, 0}, {0, c} },
                             { {0, 0}, {c, c} },
                             { {c, c}, {0, 0} } };
    for (int i = 0; i < 4; ++i)
        if (s.getPoint(corners[i][0][0], corners[i][0][1]) == 0
         && s.getPoint(corners[i][0][0], corners[i][0][1]) == 2)
            return (corners[i][0][1] + s.size * corners[i][0][0]);

    // Any corner
    for (int i = 0; i < 4; ++i)
        if (s.getPoint(corners[i][0][0], corners[i][0][1]) == 0)
            return (corners[i][0][1] + s.size * corners[i][0][0]);

    // Any side ( == any remaining move )
    return validActions[0];
}
