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
                // State(State& s) : size(s.size), state(s.state), raw(s.raw) {};
                int getPoint(int i, int j) const // Matrix indexing
                { return raw[j + i*size]; }
                int getState() const
                { return state; };

                void move(int i, int j, int value);
                void move(int position, int value);

                // setState(int)
                // setState(std::vector<int>&)
                const int size;
            protected:
                int state;
                std::vector<int> raw;
        };

        TicTacToeCMP(TicTacToeTransitionKernel const * kernel);
        State currentState;

        std::vector<double> features(const State& s) const;
        std::vector<double> features() const { return features(currentState); };

        int winner();

        void move(int position, int value);
        void move(int i, int j, int value); // Matrix indexing
        void resetState();

        void printState();

        /************ Feature indices ************/
        static const int    FEATURE_SINGLETS_X = 0;
        static const int    FEATURE_DOUBLETS_X = 1;
        static const int    FEATURE_TRIPLETS_X = 2;
        static const int FEATURE_CROSSPOINTS_X = 3;
        static const int     FEATURE_CORNERS_X = 4;
        static const int       FEATURE_FORKS_X = 5;

        static const int    FEATURE_SINGLETS_O = 6;
        static const int    FEATURE_DOUBLETS_O = 7;
        static const int    FEATURE_TRIPLETS_O = 8;
        static const int FEATURE_CROSSPOINTS_O = 9;
        static const int     FEATURE_CORNERS_O = 10;
        static const int       FEATURE_FORKS_O = 11;

        static const int FEATURE_RAW           = 12;
        /******************************************/

    private:
        const int actions;
        const int size;

};

#endif
