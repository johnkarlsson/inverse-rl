#ifndef TICTACTOETRANSITIONKERNEL_H
#define TICTACTOETRANSITIONKERNEL_H

#include "../TransitionKernel.h"
#include <set>

class TicTacToeTransitionKernel
    : public TransitionKernel
{
    public:
        TicTacToeTransitionKernel(int size);

        double getTransitionProbability(const int s,
                                        const int a,
                                        const int s2) const;

        std::set<int> getValidActions(const int s) const;

        std::vector< std::pair<state, probability> >
            getTransitionProbabilities(const int s, const int a) const;

        const int size;
        static int pointValue(int s, int p);
        static int pointValue(int s, int i, int j, int size);
        static int successor(int state, int point, int player);
        static int successor(int s, int i, int j, int size, int player);
        static int nOccupied(int s, int size);
        static int player(int s, int size);
};

#endif
