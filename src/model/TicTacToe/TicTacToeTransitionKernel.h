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

        // std::vector< std::pair<state, probability> >
        //     getTransitionProbabilities(const int s, const int a) const;

        const int size;
};

#endif
