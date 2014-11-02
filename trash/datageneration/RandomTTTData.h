#ifndef RANDOMTTTDATA_H
#define RANDOMTTTDATA_H

#include <iostream>

#include "../model/TicTacToe/TicTacToeTransitionKernel.h"
#include "../model/TicTacToe/TicTacToeCMP.h"
#include "../model/TicTacToe/RandomTTTPolicy.h"

using namespace std;

class RandomTTTData
{
    public:
        static void fun()
        {
            TicTacToeTransitionKernel cmpKernel = TicTacToeTransitionKernel(3);
            TicTacToeCMP tttCmp(&cmpKernel);
            //OptimalTTTPolicy policy(&tttCmp);
            RandomTTTPolicy policy(&tttCmp);

            int moves = 0;
            while (1)
            {
                auto checkWin = [&tttCmp, &moves]() mutable
                {
                    auto features = tttCmp.features();
                    int win = 0;
                    if (features[TicTacToeCMP::FEATURE_TRIPLETS_X] > 0)
                        win = 1;
                    else if (features[TicTacToeCMP::FEATURE_TRIPLETS_O] > 0)
                        win = 2;

                    ++moves;
                    if (win || moves == 9)
                    {
                        if (win)
                            cout << " *** PLAYER " << (win == 1 ? "X" : "O") << " WINS *** " << endl;
                        else
                            cout << " *** TIE *** " << endl;
                        // test_tictactoecmp_print(tttCmp);
                        tttCmp.printState();
                        tttCmp.resetState();
                        moves = 0;
                        char cont;
                        cout << "Continue? y / n: ";
                        cin >> cont;
                        if (cont == 'n' || cont == 'N')
                            return false;
                    }

                    return true;
                };

                tttCmp.move( policy.action(tttCmp.currentState) , 1);
                if (!checkWin())
                    return;

                // test_tictactoecmp_print(tttCmp);
                tttCmp.printState();

                tttCmp.move( policy.action(tttCmp.currentState) , 2);
                if (!checkWin())
                    return;
            }
        } 
    private:
};

#endif
