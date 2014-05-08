#ifndef RANDOMTTTPOLICY_H
#define RANDOMTTTPOLICY_H

#include "TicTacToeCMP.h"
#include "../Policy.h"

#include <random>

// Bare minimum class.
class RandomTTTPolicy // Plays 3 in a row or random (one step lookahead)
    : public Policy
{
    public:
        RandomTTTPolicy(TicTacToeCMP const * _cmp, bool _playWin = true);
        int action(TicTacToeCMP::State s);
        std::vector<std::pair<int,double>> probabilities(int s);
    private:
        TicTacToeCMP const * cmp;
        std::mt19937 gen;
        bool playWin;
};

#endif

