#include "../TransitionKernel.h"
#include "TicTacToeTransitionKernel.h"
#include "TicTacToeCMP.h"

#include <vector>
#include <set>
#include <cmath>
#include <stdexcept>
#include <assert.h>

int TicTacToeTransitionKernel::pointValue(int s, int i, int j, int size)
{
    return TicTacToeTransitionKernel::pointValue(s, j + i*size);
}

int TicTacToeTransitionKernel::pointValue(int s, int p)
{
    return (((3 << (2*p)) & s) >> (2*p));
}

int TicTacToeTransitionKernel::successor(int state, int i, int j, int size,
                                                int player)
{
    return TicTacToeTransitionKernel::successor(state, j + i*size, player);
}

int TicTacToeTransitionKernel::successor(int state, int point, int player)
{
    assert(TicTacToeTransitionKernel::pointValue(state, point) == 0);
    return (state + (player << (2*point)));
}

int TicTacToeTransitionKernel::nOccupied(int s, int size)
{
    int count = 0;
    for (int p = 0; p < size*size; ++p)
        if (pointValue(s,p) != 0)
            ++count;
    return count;
}

int TicTacToeTransitionKernel::player(int s, int size)
{
    return (((nOccupied(s, size) % 2) == 0) ? 1 : 2);
}

std::set<action> TicTacToeTransitionKernel::getValidActions(const state s) const
{
    std::set<action> output;

    // Add actions for all non-occupied points
    for (action a = 0; a < actions; ++a)
        if (pointValue(s, a) == 0)
            output.insert(a);
    return output;
}

/*
int TicTacToeTransitionKernel::pointValue(int s, int p) const
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
*/

/*
int TicTacToeTransitionKernel::player(int s) const
{
    int nOccupied = 0;
    int v;
    for (int a = size*size-1; a >= 0; --a)
    {
        v = 0;
        int av = pow(3,a);
        while (s >= av)
        {
            ++v;
            s -= av;
        }
        // At this point, pointValue(s,a) == v
        if (v != 0)
            ++nOccupied;
    }

    return ((nOccupied % 2 == 0) ? 1 : 2);
}
*/

std::vector< std::pair<state, probability> >
    TicTacToeTransitionKernel::getTransitionProbabilities(const int s,
                                                          const int a) const
/*
{
    int value = player(s);
    int s2 = s + value * pow(3,a);
    return {{s2, 1}};
}
*/
{
    int value = player(s, size);
    // int s2 = s + value * pow(3,a);
    int s2 = TicTacToeTransitionKernel::successor(s, a, value);
    std::set<int> validActionsSet = getValidActions(s2);
    std::vector<int> validActions;
    std::copy(validActionsSet.begin(), validActionsSet.end(),
              std::back_inserter(validActions));

    if (validActions.size() == 0)
        return {{s2, 1}}; // Terminal state

    const int value2 = (validActions.size() % 2 == 0) ? 2 : 1;

    // Is one of the valid actions a direct win?
    for (auto a : validActions)
    {
        int state = TicTacToeTransitionKernel::successor(s2, a, value2);
        // TicTacToeCMP::State state(size, s2);
        // state.move(a, value2);
        if (nlets(TicTacToeCMP::State(size,state), size, value2) > 0)
        {
            // return {{state.getState(), 1}}; // Terminal state with a win
            return {{state, 1}}; // Terminal state with a win
        }
    }

    // Otherwise, uniformly random
    int nActions = validActions.size();
    std::vector<std::pair<int, double>> output;
    for (int a : validActions)
    {
        int state = TicTacToeTransitionKernel::successor(s2, a, value2);
        // TicTacToeCMP::State state(this->size, s2);
        // state.move(a, value2);
        //output.push_back({state.getState(), 1.0/(double)nActions});
        output.push_back({state, 1.0/(double)nActions});
    }
    return output;
}










double TicTacToeTransitionKernel::getTransitionProbability(const int s,
                                                           const int a,
                                                           const int s2) const
{
    throw std::runtime_error(
    "Undefined function TicTacToeTransitionKernel::getTransitionProbability");
}

TicTacToeTransitionKernel::TicTacToeTransitionKernel(int _size)
    : TransitionKernel(pow(3,_size*_size), _size*_size),
      size(_size)
{}

