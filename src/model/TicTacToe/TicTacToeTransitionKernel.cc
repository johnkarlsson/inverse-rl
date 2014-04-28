#include "../TransitionKernel.h"
#include "TicTacToeTransitionKernel.h"

#include <vector>
#include <set>
#include <cmath>

int pointValue(int s, int p)
{
    int v;
    for (int a = 8; a >= p; --a) // Stop at p
    {
        v = 0;
        int av = pow(3,a);
        while (s >= av)
        {
            ++v;
            s -= av;
        }
        // At this point, pointValue(s,a) == v
    }
    return v;
}

double TicTacToeTransitionKernel::getTransitionProbability(const int s,
                                                           const int a,
                                                           const int s2) const
{
    return 0.0;
}

TicTacToeTransitionKernel::TicTacToeTransitionKernel(int _size)
    : TransitionKernel(pow(3,_size*_size), _size*_size),
      size(_size)
{}

std::set<action> TicTacToeTransitionKernel::getValidActions(const state s) const
{
    std::set<action> output;

    // Add actions for all non-occupied points
    for (action a = 0; a < actions; ++a)
        if (pointValue(s, a) == 0)
            output.insert(a);
    return output;
}
