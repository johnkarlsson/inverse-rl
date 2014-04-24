#ifndef TICTACTOECMP_H
#define TICTACTOECMP_H

#include "../DiscreteCMP.h"
#include "TicTacToeTransitionKernel.h"
#include <vector>
#include <cmath>
#include <stdexcept>

class TicTacToeCMP : public DiscreteCMP
{
    public:

        struct State
        {
            friend class TicTacToeCMP;
            public:
                State(int _size) : size(_size), state(0),
                                   raw(std::vector<int>(_size*_size,0)) {};
                int getPoint(int i, int j) const // Matrix indexing
                { return raw[j + i*size]; }
                int getState() const
                { return state; };
                // setState(int)
                // setState(std::vector<int>&)
                const int size;
            protected:
                int state;
                std::vector<int> raw;
        };

        TicTacToeCMP(TicTacToeTransitionKernel const * kernel);
        State currentState;

        std::vector<double> features(State& s);
        std::vector<double> features() { return features(currentState); };

        void move(int i, int j, int value); // Matrix indexing

        void printState();

    private:
        const int actions;
        const int size;

};

#endif
