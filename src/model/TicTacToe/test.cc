#include <iostream>

#include "TicTacToeTransitionKernel.h"
#include "TicTacToeCMP.h"
#include "TicTacToeMDP.h"
#include "RandomTTTPolicy.h"
#include "OptimalTTTPolicy.h"

#include "../../algorithm/LSPI.h"

using std::cout;
using std::endl;
using std::cin;

void test_tictactoecmp_print(TicTacToeCMP& tttCmp);
std::pair<double,double> optimal_vs_random();

vector<double> wGlobal; // For testLstdqOptimalpolicy and Play for TTT

void test_tictactoetransitionkernel()
{
    cout << "*** Testing TicTacToeTransitionKernel..." << endl;
    TicTacToeTransitionKernel cmpKernel = TicTacToeTransitionKernel(3);
    DiscreteCMP cmp(&cmpKernel);
    // Occupied: 5, 7, 3, 0
    // ==> Valid actions: 1,2,4,6,8
    int s = pow(3,5)*2 + pow(3,7)*2 + pow(3,3)*1 + pow(3,0)*1;
    set<action> validActions = cmp.kernel->getValidActions(s);
    cout << "Occupied: 0 3 5 7. Valid actions: ";
    for (action a : validActions)
        cout << a << " ";
     cout << endl;
}

void play_optimalTTTpolicy()
{
    TicTacToeTransitionKernel cmpKernel = TicTacToeTransitionKernel(3);
    TicTacToeCMP tttCmp(&cmpKernel);
    // OptimalTTTPolicy policy(&tttCmp);
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
                test_tictactoecmp_print(tttCmp);
                tttCmp.resetState();
                moves = 0;
            }
        };

        tttCmp.move( policy.action(tttCmp.currentState) , 1);
        checkWin();

        test_tictactoecmp_print(tttCmp);

        double sum = 0;
        auto features = tttCmp.features(); 
        int k = features.size();
        for (int i = 0; i < k; ++i)
            sum += features[i]*wGlobal[i];
        cout << "Value for X: " << sum << endl;

        int i,j;
        cin >> i >> j;

        tttCmp.move(i-1, j-1, 2);
        checkWin();
    }
}

void test_lstdq_optpolicy()
{
    TicTacToeTransitionKernel cmpKernel = TicTacToeTransitionKernel(3);
    TicTacToeCMP tttCmp(&cmpKernel);
    RandomTTTPolicy randomPolicy(&tttCmp);
    OptimalTTTPolicy optimalPolicy(&tttCmp);

    const int n = 10000; // O(n)
    const int k = tttCmp.features().size();
    const double gamma = 1.0;

    vector<double> phi(k*n);
    vector<double> td(n*k);
    vector<double> b(k);

    int player = 2;
    for (int i = 0; i < n; ++i)
    {
        if (tttCmp.winner() != 0)
        {
            tttCmp.resetState();
            // player = 2; 
        }

        // Play 2 moves (X + O)
        bool terminal = false;

        do 
        {
            tttCmp.move(randomPolicy.action(tttCmp.currentState), player);
            player = (player == 2) ? 1 : 2;
            if (tttCmp.winner())
                terminal = true;
        } while (!terminal && player == 2);
        /*
        for (int i = 0; i < 2; ++i)
        {
            player = (player == 2) ? 1 : 2;
            // if (player == 2)
                tttCmp.move(randomPolicy.action(tttCmp.currentState), player);
            // else
            //     tttCmp.move(optimalPolicy.action(tttCmp.currentState), player);
            if (tttCmp.winner())
            {
                terminal = true;
                break;
            }
        }
        */

        TicTacToeCMP::State s2(tttCmp.currentState);

        auto features = tttCmp.features(s2.getState());
        // features[TicTacToeCMP::FEATURE_FORKS_X] = 0;
        // features[TicTacToeCMP::FEATURE_FORKS_O] = 0;
        // normalize(features);

        if (!terminal)
            s2.move(optimalPolicy.action(s2), 1);
        auto featuresPi = tttCmp.features(s2.getState());
        // featuresPi[TicTacToeCMP::FEATURE_FORKS_X] = 0;
        // featuresPi[TicTacToeCMP::FEATURE_FORKS_O] = 0;
        // normalize(featuresPi);

        double r;
        const double rewards[] = {0, 1, -1000, 0};
        int win = tttCmp.winner();
        r = rewards[win];
        for (int j = 0; j < k; ++j)
        {
            // phi[i * k + j] = features[j];
            phi[j * n + i] = features[j];
            if (terminal)
                td[i * k + j] = features[j];
            else
                td[i * k + j] = (features[j] - gamma * featuresPi[j]);
            if (r != 0)
                b[j] += r * features[j];
        }
    }

    cout << "..." << endl;

    vector<double> w = LSPI::solve(k, n, phi, td, b);

    cout << "LSTDQ(optimal policy) w* = ";
    for (int i = 0; i < k; ++i)
        cout << w[i] << " ";
    cout << endl;

    tttCmp.resetState();
    tttCmp.move(2,2, 1);
    tttCmp.move(2,1, 1);
    tttCmp.move(1,2, 1);
    test_tictactoecmp_print(tttCmp);
    double sum = 0;
    auto features = tttCmp.features(); 
    for (int i = 0; i < k; ++i)
        sum += features[i]*w[i];
    cout << "Value: " << sum << endl;

    wGlobal = w;
}

void test_tictactoecmp_print(TicTacToeCMP& tttCmp)
{
    tttCmp.printState();

    OptimalTTTPolicy policy(&tttCmp);

    set<action> validActions =
        tttCmp.kernel->getValidActions(tttCmp.currentState.getState());
    cout << "Valid actions: ";
    for (action a : validActions)
        cout << a << " ";
    if (validActions.size() > 0)
        cout << " (" << policy.action(tttCmp.currentState) << " optimal)";
    cout << endl;

    cout << "Features X(s,d,t,x,c,f) O(s,d,t,x,c,f) Raw(1-9): ";
    vector<double> w = tttCmp.features();

    for (double wi : w)
        cout << wi << " ";
    cout << endl;
}

void test_tictactoecmp()
{
    cout << "*** Testing TicTacToeCMP..." << endl;
    TicTacToeTransitionKernel cmpKernel = TicTacToeTransitionKernel(3);
    TicTacToeCMP tttCmp(&cmpKernel); //

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 2; ++j)
            tttCmp.move(i,j, 1);
    tttCmp.move(0,2, 2);
    tttCmp.move(1,2, 2);
    tttCmp.move(2,2, 2);
    test_tictactoecmp_print(tttCmp);


    tttCmp.resetState();
    tttCmp.move(1,2, 1);
    tttCmp.move(1,1, 2);
    test_tictactoecmp_print(tttCmp);
    tttCmp.move(0,1, 1);
    tttCmp.move(0,2, 2);
    tttCmp.move(0,0, 1);
    test_tictactoecmp_print(tttCmp);
    tttCmp.move(2,0, 1);
    tttCmp.move(1,0, 2);
    tttCmp.move(2,2, 1);
    test_tictactoecmp_print(tttCmp);

    tttCmp.resetState();
    test_tictactoecmp_print(tttCmp);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            tttCmp.move(i,j, 1);
    tttCmp.move(2,1, 2);
    test_tictactoecmp_print(tttCmp);

    tttCmp.resetState();
    tttCmp.move(1,0, 1);
    tttCmp.move(0,2, 1);
    test_tictactoecmp_print(tttCmp);

    // Fork 1
    tttCmp.resetState();
    tttCmp.move(2,2, 1);
    tttCmp.move(2,1, 1);
    tttCmp.move(1,2, 1);
    test_tictactoecmp_print(tttCmp);

    // Fork 2
    tttCmp.resetState();
    tttCmp.move(0,0, 1);
    tttCmp.move(0,2, 1);
    tttCmp.move(2,0, 1);
    test_tictactoecmp_print(tttCmp);

    // Fork 3
    tttCmp.resetState();
    tttCmp.move(2,0, 1);
    tttCmp.move(1,1, 1);
    tttCmp.move(1,2, 1);
    tttCmp.move(2,2, 1);
    test_tictactoecmp_print(tttCmp);

    // Fork 4
    tttCmp.resetState();
    tttCmp.move(2,0, 1);
    tttCmp.move(1,1, 1);
    tttCmp.move(1,2, 1);
    tttCmp.move(2,2, 1);
    tttCmp.move(0,2, 2);
    test_tictactoecmp_print(tttCmp);
    std::pair<double,double> results = optimal_vs_random();
    cout << "Optimal vs. random: X: " << results.first << " O: " << results.second << endl;
}

std::pair<double, double> optimal_vs_random()
{
    TicTacToeTransitionKernel cmpKernel = TicTacToeTransitionKernel(3);
    TicTacToeCMP tttCmp(&cmpKernel);
    OptimalTTTPolicy p1(&tttCmp);
    RandomTTTPolicy p2(&tttCmp);
    int p1wins = 0;
    int p2wins = 0;
    // auto winner = [&tttCmp]() mutable
    // {
    //     auto features = tttCmp.features();
    //     if (features[TicTacToeCMP::FEATURE_TRIPLETS_X] > 0)
    //         return 1;
    //     else if (features[TicTacToeCMP::FEATURE_TRIPLETS_O] > 0)
    //         return 2;
    //     else if (tttCmp.kernel->getValidActions(tttCmp.currentState.getState()).size() == 0)
    //         return 3;
    //     else
    //         return 0;
    // };

    auto checkWin = [&p1wins, &p2wins, &tttCmp]() mutable
    {
        int win = tttCmp.winner();
        if (win != 0)
        {
            tttCmp.resetState();
            switch (win)
            {
                case 1: ++p1wins; break;
                case 2: ++p2wins; break;
            }
        }
    };

    int i = 0;
    for (i = 0; i < 10000; ++i)
    {
        tttCmp.move( p1.action(tttCmp.currentState) , 1 );
        int win = tttCmp.winner();
        checkWin();
        if (win != 0)
            continue; // Always X first
        tttCmp.move( p2.action(tttCmp.currentState) , 2 );
        checkWin();
    }

    return std::pair<double,double> ( ((double) p1wins) / i , ((double) p2wins) / i );
}

void test_generateTTT()
{
    TicTacToeTransitionKernel kernel = TicTacToeTransitionKernel(3);
    TicTacToeCMP cmp(&kernel);
    TicTacToeMDP mdp(&cmp, true); // True rewards
    RandomTTTPolicy policyRandom(&cmp);
    OptimalTTTPolicy policyOptimal(&cmp);
    // vector<Policy*> policies = {&policyRandom, &policyOptimal};
    vector<Policy*> policies = { &policyRandom };
    vector<Demonstration> demonstrations
        = generateDemonstrations(mdp, policies, -1, 10, 0);

    // vector<double> optWeights(cmp.nFeatures());
    auto optWeights = LSPI::lstdq(demonstrations, policyOptimal, mdp);
    cout.precision(numeric_limits<double>::digits10 - 12);
    cout << "Optimal weights X(s,d,t,x,c,f) O(s,d,t,x,c,f) : " << endl;
    for (double d : optWeights)
        cout << fixed << d << "\t";
    cout << endl;
    cout << "LSPI weights: " << endl;
    DeterministicPolicy lspiPolicy = LSPI::lspi(demonstrations, mdp);
    auto lspiWeights = lspiPolicy.getWeights();
    for (double d : lspiWeights)
        cout << fixed << d << "\t";
    cout << endl;

    const int maxPrints = 0;
    int prints = 0;
    for (Demonstration d : demonstrations)
    {
        if (++prints > maxPrints)
            break;
        /*
        for (Transition tr : d)
        {
            int s = tr.s;
            cmp.printState(TicTacToeCMP::State(cmp.size, s));
        }
        */
        int s = d.back().s;
        cmp.printState(TicTacToeCMP::State(cmp.size, s));
        auto phi = cmp.features(s);
        cout << "Value X(s,d,t,x,c,f) O(s,d,t,x,c,f) = " << endl << "\t";
        for (double f : phi)
            cout << fixed << f << "\t";
        cout << endl << " *\t";
        double q = std::inner_product(phi.begin(), phi.end(),
                                      optWeights.begin(), 0.0);
        for (double d : optWeights)
            cout << fixed << d << "\t";
        cout << endl;
        cout << " =\t" << q << endl;
    }
    // cout << "Global count: " << globalCount << endl;
    // cout << "Global count2: " << globalCount2 << endl;
}
