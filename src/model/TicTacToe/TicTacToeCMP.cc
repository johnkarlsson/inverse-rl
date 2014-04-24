#include "TicTacToeCMP.h"
#include "TicTacToeTransitionKernel.h"
#include "../DiscreteCMP.h"

#include <cmath>
#include <vector>
#include <iostream>

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
    if (! ( value == 1 || value == 2) )
        throw std::invalid_argument(
                "State.move() called with value \\notin {1,2}");
    if (i < 0 || i > size || j < 0 || j > size)
        throw std::invalid_argument(
                "State.move() called with invalid position.");
    if (currentState.getPoint(i,j) != 0)
        throw std::invalid_argument(
                "State.move() called with occupied position.");

    int position = j + size*i;

    currentState.state += value * pow(3,position);
    currentState.raw[position] = value;
}

// Counts the number of nlets (singlets, doublets, triplets)
int nlets(TicTacToeCMP::State& s, int n, int player)
{
    int found = 0;
    bool diagsValid[2] = {true,true};
    int  diagsCount[2] = {0,0};

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
        if (rowValid && rowCount >= n)
            ++found;
        if (colValid && colCount >= n)
            ++found;

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
        if (diagsValid[d] && diagsCount[d] >= n)
            ++found;

    return found;
}

std::vector<double> TicTacToeCMP::features(State& s)
{
    const int nFeatures = 
        2*size // number of nlets
        + 1;   // something

    std::vector<double> phi(nFeatures, 0);

    int i = 0;
    // #(singlets, doublets, ..., nlets) for each player
    for (int player = 1; player <= 2; ++player)
        for (int nlet = 1; nlet <= size; ++nlet)
            phi[i++] = nlets(currentState, nlet, player);

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
