#include "TicTacToeCMP.h"
#include "TicTacToeTransitionKernel.h"
#include "../DiscreteCMP.h"

#include <cmath>
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
    // cout << "\t ^^^^^^^ " << endl;
    
    //         _______
    //        | X O X |
    //        | O X X |
    //        | X O X |
    //         ^^^^^^^
}
