#ifndef TICTACTOECMP_H
#define TICTACTOECMP_H

#include "../DiscreteCMP.h"
#include "TicTacToeTransitionKernel.h"
#include <vector>
#include <cmath>
#include <stdexcept>

class TicTacToeCMP
    : public DiscreteCMP
{
    public:

        struct State
        {
            friend class TicTacToeCMP;
            public:
                State(int _size) : size(_size), state(0) {};
                State(int _size, int _state) : size(_size), state(_state) {};
                inline int getPoint(int i, int j) const // Matrix indexing
                {
                    return TicTacToeTransitionKernel::pointValue(state, i, j,
                                                                 size);
                }
                inline int getPoint(int a) const
                {
                    return TicTacToeTransitionKernel::pointValue(state, a);
                }

                int getState() const
                { return state; };

                void move(int i, int j, int value);
                void move(int position, int value);

                void setState(int s, bool invert = false)
                {
                    state = s;
                    if (invert)
                    {
                        int invertedState = 0;
                        for (int p = 0; p < size*size; ++p)
                        {
                            int v = TicTacToeTransitionKernel::pointValue(s, p);
                            if (v != 0)
                            {
                                int vi = (v == 1) ? 2 : 1;
                                invertedState = TicTacToeTransitionKernel
                                            ::successor(invertedState, p, vi);
                            }
                        }
                        state = invertedState;
                    }
                }
                const int size;
            protected:
                int state;
                // std::vector<int> raw;
        };

        TicTacToeCMP(TicTacToeTransitionKernel const * kernel);
        State currentState;

        using DiscreteCMP::features;

        // std::vector<double> features(const State& s) const;
        std::vector<double> features(const State& s) const
        {
            return features(s.getState());
        }
        std::vector<double> features(int s) const;
        std::vector<double> features() const
        {
            throw std::runtime_error("features() unsupported.");
            // return features(currentState);
        };

        std::vector<double> features(int s, int a) const;

        int nFeatures() const;

        int winner(const State& state) const;
        int winner() const;
        bool isTerminal(int s) const;

        void move(int position, int value);
        void move(int i, int j, int value); // Matrix indexing
        void resetState();

        void printState();
        static void printState(int s, int size);
        static void printState(TicTacToeCMP::State state);

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

        static const int        FEATURE_CENTER = 12;

        static const int           FEATURE_RAW = 13;
        /******************************************/

        const int size;
        const int actions;
    private:
        std::vector<double> _features(const State& s) const;
};

int nlets(const TicTacToeCMP::State& s, int n, int player,
          bool crosspoints = false);

#endif
