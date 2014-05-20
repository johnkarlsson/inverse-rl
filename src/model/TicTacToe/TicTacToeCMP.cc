#include "TicTacToeCMP.h"
#include "TicTacToeTransitionKernel.h"
#include "../DiscreteCMP.h"

#include <cmath>
#include <vector>
#include <iostream>
#include <sstream>

using std::cout;
using std::endl;

TicTacToeCMP::TicTacToeCMP(TicTacToeTransitionKernel const * tttKernel)
    : DiscreteCMP(tttKernel),
      currentState(tttKernel->size),
      size(tttKernel->size),
      actions(tttKernel->actions)
{}

void TicTacToeCMP::move(int position, int value)
{
    currentState.move(position, value);
}

void TicTacToeCMP::move(int i, int j, int value)
{
    currentState.move(i, j, value);
}

void TicTacToeCMP::State::move(int position, int value)
{
    int i = position / size;
    int j = position - i * size;

    move(i, j, value);
}

int TicTacToeCMP::winner() const
{
    return winner(currentState);
}

bool TicTacToeCMP::isTerminal(int s) const
{
    return (winner(TicTacToeCMP::State(size, s)) != 0);
}

// int TicTacToeCMP::winner()
int TicTacToeCMP::winner(const State& state) const
{
    /*
    auto features = this->features(state);
    if (features[TicTacToeCMP::FEATURE_TRIPLETS_X] > 0)
        return 1;
    else if (features[TicTacToeCMP::FEATURE_TRIPLETS_O] > 0)
        return 2;
    else if (kernel->getValidActions(state.getState()).size() == 0)
        return 3;
    else
        return 0;
    */
    if (nlets(state, this->size, 1) > 0)
        return 1;
    else if (nlets(state, this->size, 2) > 0)
        return 2;
    else
        for (int a = 0; a < size*size; ++a)
            if (state.getPoint(a) == 0)
                return 0; // Game on
    return 3; // All tiles filled, tie.
}

void TicTacToeCMP::State::move(int i, int j, int value)
{
    std::ostringstream os;
    os << "(" << i << "," << j << ")";

    if (! ( value == 1 || value == 2) )
        throw std::invalid_argument(
            "State.move() called with value \\notin {1,2}");
    if (i < 0 || i > size || j < 0 || j > size)
        throw std::invalid_argument(
            "State.move() called with invalid position " + os.str() + ".");
    if (getPoint(i,j) != 0)
        throw std::invalid_argument(
            "State.move() called with occupied position " + os.str() + ".");

    state = TicTacToeTransitionKernel::successor(state, i, j, size, value);
    /*
    int position = j + size*i;

    state += value * pow(3,position);
    raw[position] = value;
    */
}

void TicTacToeCMP::resetState()
{
    currentState.state = 0;
    // for (int p = 0; p < size*size; ++p)
    //     currentState.raw[p] = 0;
}

inline int successor(int state, int i, int j, int player)
{
    static const int size = 3;
    return TicTacToeTransitionKernel::successor(state, i, j, size, player);
}

int triplets(int s, int player)
{
    static int nTriplets = 8;
    static std::vector<std::vector<int>> tr(2, std::vector<int>(nTriplets, 0));
    if (tr[0][0] == 0)
    {
        for (int p = 0; p <= 1; ++p)
        {
                int t = 0;
                // Diag 1
                tr[p][t] = successor(tr[p][t], 0, 0, p+1);
                tr[p][t] = successor(tr[p][t], 1, 1, p+1);
                tr[p][t] = successor(tr[p][t], 2, 2, p+1);
                ++t;

                // Diag 2
                tr[p][t] = successor(tr[p][t], 0, 2, p+1);
                tr[p][t] = successor(tr[p][t], 1, 1, p+1);
                tr[p][t] = successor(tr[p][t], 2, 0, p+1);
                ++t;

                for (int c = 0; c < 3; ++c)
                {
                    for (int i = 0; i < 3; ++i)
                        tr[p][t] = successor(tr[p][t], i, c, p+1);
                    ++t;
                }
                for (int r = 0; r < 3; ++r)
                {
                    for (int i = 0; i < 3; ++i)
                        tr[p][t] = successor(tr[p][t], r, i, p+1);
                    ++t;
                }
        }
        // for (int p = 0; p <= 1; ++p)
        //     for (int t = 0; t < nTriplets; ++t)
        //         TicTacToeCMP::printState(tr[p][t], 3);
    }

    int count = 0;
    for (int triplet : tr[player-1])
    {
        if ((s & triplet) == triplet)
            ++count;
    }
    return count;
}


/**
 * For 2000 demonstrations, one LSPI call renders ~80M invocations to this
 * method. Fastest way around it is a lookup table, for which only 65 000
 * invocations are needed (but inefficient space ~16 MB data).
 *
 * Demonstrations  Calls   Without lookup
 *           4000  65 000
 *           2000  65 000  80 000 000
 *            500  64 000
 *            100  51 000
 *            100  42 000
 */
char nlets_memoized[2 << (9*2 + 4)]; // 16 MB
int _nlets(const TicTacToeCMP::State& s, int n, int player, bool crosspoints);
// Counts the number of nlets (singlets, doublets, triplets)
int nlets(const TicTacToeCMP::State& s, int n, int player, bool crosspoints)
{
    static const int ss = 2 * s.size * s.size;
    int lookup = s.getState()
                 + (n << ss)
                 + (player << (ss+2))
                 + (crosspoints << (ss+3));
    int nLookup = nlets_memoized[lookup];
    if (nLookup == 0) // domain is >= 0; store value + 1 to let 0 indicate miss
    {
        nLookup = 1 + _nlets(s, n, player, crosspoints);
        nlets_memoized[lookup] = nLookup;
    }
    return (nLookup - 1);
}

int _nlets(const TicTacToeCMP::State& s, int n, int player, bool crosspoints)
{
    if (n == 3 && s.size == 3) // This optimiziation turned out rather useless.
        return triplets(s.getState(), player); // But it's not worse...

    int found = 0;
    bool diagsValid[2] = {true,true};
    int  diagsCount[2] = {0,0};

    std::vector<int> crossCounts(s.size*s.size,0);

    auto tagFound = [=, &crossCounts, &found, &s](int l, bool row) mutable
    {
        ++found;
        for (int k = 0; k < s.size; ++k)
            if (row)
                crossCounts[k + l*s.size]++;
            else // col
                crossCounts[l + k*s.size]++;
    };

    for (int i = 0; i < s.size; ++i)
    {
        int rowCount = 0;
        int colCount = 0;
        bool rowValid = true;
        bool colValid = true;
        for (int j = 0; j < s.size; ++j)
        {
            if (s.getPoint(i,j) == player)
                ++rowCount;
            else if (s.getPoint(i,j) != 0)
                rowValid = false;
            if (s.getPoint(j,i) == player)
                ++colCount;
            else if (s.getPoint(j,i) != 0)
                colValid = false;
        }
        if (rowValid && rowCount == n)
            tagFound(i, true); // ++found;
        if (colValid && colCount == n)
            tagFound(i, false); // ++found;

        // Diagonals
        for (int d = 0; d < 2; ++d)
        {
            int j = ( d ) * (s.size - i - 1) + (1 - d) * i;

            if (s.getPoint(i,j) == player)
                ++diagsCount[d];
            else if (s.getPoint(i,j) != 0)
                diagsValid[d] = false;
        }
    }

    for (int d = 0; d < 2; ++d)
    {
        if (diagsValid[d] && diagsCount[d] == n)
        {
            ++found;
            for (int i = 0; i < s.size; ++i)
            {
                int j = ( d ) * (s.size - i - 1) + (1 - d) * i;
                crossCounts[j + i*s.size]++;
            }
        }
    }

    // for (int d = 0; d < 2; ++d)
    //     if (diagsValid[d] && diagsCount[d] == n)
    //         ++found;

    if (!crosspoints)
        return found;
    else
    {
        int nCrosspoints = 0;
        for (int i = 0; i < s.size; ++i)
            for (int j = 0; j < s.size; ++j)
                // only empty crosspoints count
                if (s.getPoint(i,j) == 0 && crossCounts[j + i*s.size] >= 2)
                    ++nCrosspoints;
        return nCrosspoints;
    }
}

int TicTacToeCMP::nFeatures() const
{
    return
          2 * size    // number of nlets
        + 2           // crosspoints
     // + size*size   // raw data
        + 2           // corners per player
        + 2           // forks per player
        + 1;          // center occupation

    // TODO: Add bias (1) feature ?
}

/*
 * 85% increase (ontopof features(s) optimization)
 */
static std::vector<double> memoized_action_features[1 << (9*2 + 3)];
std::vector<double> TicTacToeCMP::features(int s, int a) const
{
    int ai = a << (9*2);
    int mi = s + ai;
    if (memoized_action_features[mi].size() == 0)
        memoized_action_features[mi] = DiscreteCMP::features(s,a);
    return memoized_action_features[mi];
}

/*
 * 21% increase
 */
static std::vector<double> memoized_features[1 << (9*2)]; // hardcoded 3x3 board
// std::vector<double> TicTacToeCMP::features(const State& state) const
std::vector<double> TicTacToeCMP::features(int s) const
{
    // int s = state.getState();
    if (memoized_features[s].size() == 0)
        memoized_features[s] = _features(TicTacToeCMP::State(size, s));
        // memoized_features[s] = _features(state);
    return memoized_features[s];
}

std::vector<double> TicTacToeCMP::_features(const State& s) const
{

    const int _nFeatures = nFeatures();

    std::vector<double> phi(_nFeatures, 0);

    int i = 0;
    for (int player = 1; player <= 2; ++player)
    {
        // #(singlets, doublets, ..., nlets) for each player
        for (int nlet = 1; nlet <= size; ++nlet)
            phi[i++] = nlets(s, nlet, player);
        // crosspoints for each player
        phi[i++] = nlets(s, 1, player, true);
        // corners per player
        int nCorners = 0;
        const int coords[2] = {0, s.size - 1};
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b)
                if (s.getPoint(coords[a], coords[b]) == player)
                    ++nCorners;
        phi[i++] = nCorners;
        // forks per player
        phi[i++] = (nlets(s, 2, player, false) - nlets(s, 2, player, true)) / 2;
    }

    // center occupation
    int c = (int)(s.size/2);
    switch (s.getPoint(c,c))
    {
        case 0: phi[i++] = 0; break;
        case 1: phi[i++] = 1; break;
        case 2: phi[i++] =-1; break;
    }

    // raw board
    // for (int k = 0; k < size; ++k)
    //     for (int l = 0; l < size; ++l)
    //         phi[i++] = s.getPoint(k, l);

    return phi;
}

void TicTacToeCMP::printState(int s, int size)
{
    printState(TicTacToeCMP::State(size, s));
}

void TicTacToeCMP::printState()
{
    TicTacToeCMP::printState(currentState);
}

void TicTacToeCMP::printState(TicTacToeCMP::State state)
{
    auto printChars = [&,state](char c)
    { 
        cout << "\t  " << c;
        for (int s = 0; s < 2*state.size; ++s)
            cout << c;
        cout << endl;
    };

    printChars('_');

    for (int i = 0; i < state.size; ++i)
    {
        cout << "\t | ";
        for (int j = 0; j < state.size; ++j)
        {
            switch (state.getPoint(i,j))
            {
                case 0:  cout << "  "; break;
                case 1:  cout << "X "; break;
                case 2:  cout << "O "; break;
                default: cout << "? ";
            }
        }
        cout << "|" << endl;
    }

    printChars('^');
}
