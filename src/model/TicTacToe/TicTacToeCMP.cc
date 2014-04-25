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
      actions(tttKernel->actions),
      currentState(tttKernel->size),
      size(tttKernel->size)
{}

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

    int position = j + size*i;

    state += value * pow(3,position);
    raw[position] = value;
}

void TicTacToeCMP::resetState()
{
    currentState.state = 0;
    for (int p = 0; p < size*size; ++p)
        currentState.raw[p] = 0;
}

// Counts the number of nlets (singlets, doublets, triplets)
int nlets(const TicTacToeCMP::State& s, int n, int player, bool crosspoints = false)
{
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

std::vector<double> TicTacToeCMP::features(const State& s) const
{
    const int nFeatures = 
        2*size        // number of nlets
        + 2           // crosspoints
        + size*size   // raw data
        + 2           // corners per player
        + 2           // forks per player
        + 1;          // center occupation

    std::vector<double> phi(nFeatures, 0);

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

    // raw board
    for (int k = 0; k < size; ++k)
        for (int l = 0; l < size; ++l)
            phi[i++] = s.getPoint(k, l);

    // center occupation
    int c = (int)(s.size/2);
    switch (s.getPoint(c,c))
    {
        case 0: phi[i++] = 0; break;
        case 1: phi[i++] = 1; break;
        case 2: phi[i++] =-1; break;
    }

    return phi;
}

void TicTacToeCMP::printState()
{
    auto printChars = [this](char c)
    { 
        cout << "\t  " << c;
        for (int s = 0; s < 2*size; ++s)
            cout << c;
        cout << endl;
    };

    printChars('_');

    for (int i = 0; i < size; ++i)
    {
        cout << "\t | ";
        for (int j = 0; j < size; ++j)
        {
            switch (currentState.getPoint(i,j))
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
