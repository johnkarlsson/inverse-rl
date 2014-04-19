#include <iostream>
#include "model/DiscreteCMP.h"
#include "model/DiscreteMDP.h"
#include "model/GridCMP.h"
#include "algorithm/ValueIteration.h"
#include <cfloat>
#include <limits>
#include <chrono>

using namespace std;

void test_gridcmp();
void test_discretecmp();
void test_valueiteration();
void compare_vi_qi(int cmp_size=10, double epsilon=0.001);

int main(int argc, const char *argv[])
{
    test_discretecmp();
    test_gridcmp();
    // test_valueiteration();
    compare_vi_qi(30, 10);
    return 0;
}

void test_gridcmp()
{
    cout << "*** Testing GridCMP..." << endl;
    GridCMP cmp(10); //
    double prob = cmp.getTransitionProbability(0, 1, 1);
    std::cout << "Prob: " << prob << std::endl;
}


void test_discretecmp()
{
    cout << "*** Testing DiscreteCMP..." << endl;
    DiscreteCMP cmp(10, 10); //
    cmp.setTransitionProbability(0, 0, 0, 0.7);
    double prob = cmp.getTransitionProbability(0, 0, 0);
    std::cout << "Prob: " << prob << std::endl;
}

/*
void test_valueiteration()
{
    cout << "*** Testing ValueIteration on GridCMP" << endl;

    const int cmp_size = 10;

    GridCMP cmp(cmp_size); // 10x10 grid, 100 states.
    DiscreteMDP mdp(&cmp, 0.99);
    for (int s = 0; s < cmp.states; ++s)
        mdp.setReward(s, -1);
    mdp.setReward(cmp_size - 1, 100); // top right state has reward 100, rest -1
    // Make top right state terminal
    for (int s2 = 0; s2 < cmp.states; ++s2)
        for (int a = 0; a < cmp.actions; ++a)
            cmp.setTransitionProbability(cmp_size - 1, a, s2, 0);

    ValueIteration vi(&mdp);

    cout << "Running value iteration "; flush(cout);
    for (int i = 0; i < 10; ++i)
    {
        cout << "."; flush(cout);
        // vi.init();
        vi.computeStateValues();
    }
    cout << " Done!" << endl;

    cout.precision(numeric_limits< double >::digits10 - 12);
    for (int y = 0; y < cmp_size; ++y)
    {
        for (int x = 0; x < cmp_size; ++x)
        {
            int s = x + y * cmp_size;
            // double Q_max = -DBL_MAX;
            // for (int a = 0; a < cmp.actions; ++a)
            //     if (vi.Q[s][a] > Q_max)
            //         Q_max = vi.Q[s][a];
            // cout << fixed << Q_max << " | \t";
            cout << fixed << vi.V[s] << " | \t";
        }
        cout << endl;
    }
    cout << endl;
}
*/

DiscreteMDP getGridMDP(int cmp_size)
{
    GridCMP *cmp = new GridCMP(cmp_size); // 10x10 grid, 100 states.
    DiscreteMDP mdp(cmp, 0.99);
    for (int s = 0; s < cmp->states; ++s)
        mdp.setReward(s, -1);
    mdp.setReward(cmp_size - 1, 100); // top right state has reward 100, rest -1
    // Make top right state terminal
    for (int s2 = 0; s2 < cmp->states; ++s2)
        for (int a = 0; a < cmp->actions; ++a)
            cmp->setTransitionProbability(cmp_size - 1, a, s2, 0);

    return mdp;
}

void printGridCMP_Q(ValueIteration &vi, int cmp_size, int actions)
{
    cout.precision(numeric_limits< double >::digits10 - 12);
    for (int y = 0; y < cmp_size; ++y)
    {
        for (int x = 0; x < cmp_size; ++x)
        {
            int s = x + y * cmp_size;
            double Q_max = -DBL_MAX;
            for (int a = 0; a < actions; ++a)
                if (vi.Q[s][a] > Q_max)
                    Q_max = vi.Q[s][a];
            cout << fixed << Q_max << " | \t";
        }
        cout << endl;
    }
    cout << endl;
}

void printGridCMP_V(ValueIteration &vi, int cmp_size)
{
    cout.precision(numeric_limits< double >::digits10 - 12);
    for (int y = 0; y < cmp_size; ++y)
    {
        for (int x = 0; x < cmp_size; ++x)
        {
            int s = x + y * cmp_size;
            cout << fixed << vi.V[s] << " | \t";
        }
        cout << endl;
    }
    cout << endl;
}

void compare_vi_qi(int cmp_size, double epsilon)
{
    cout << "*** Comparing timing of ValueIteration (V/Q) on GridCMP(" << cmp_size << ")" << endl;

    DiscreteMDP mdp = getGridMDP(cmp_size);

    ValueIteration vi(&mdp);

    auto t1_v = std::chrono::high_resolution_clock::now();
    vi.computeStateValues(epsilon);
    auto t2_v = chrono::high_resolution_clock::now();
    auto time_v =
        chrono::duration_cast<chrono::milliseconds>(t2_v-t1_v).count();

    auto t1_q = std::chrono::high_resolution_clock::now();
    vi.computeStateActionValues(epsilon);
    auto t2_q = chrono::high_resolution_clock::now();
    auto time_q =
        chrono::duration_cast<chrono::milliseconds>(t2_q-t1_q).count();

    cout << "t(V) = " << time_v << " ms" << endl;
    cout << "t(Q) = " << time_q << " ms" << endl;

    delete(mdp.cmp);
}
