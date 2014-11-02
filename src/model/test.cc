#include "DirichletPolicyPosterior.h"
#include "random_mdp/RandomTransitionKernel.h"
#include "random_mdp/RandomCMP.h"
#include "random_mdp/RandomMDP.h"
#include "Policy.h"

#include "../util.h"

void test_dirichletPolicyPosterior()
{
    const double gamma = 0.5;
    const int states = 6;
    const int actions = 10;
    RandomTransitionKernel kernel(states, actions);
    RandomCMP cmp(&kernel);
    RandomMDP mdp(&cmp, gamma);
    RandomPolicy randomPolicy(mdp.cmp);
    auto demonstrations =
        generateDemonstrations(mdp, {&randomPolicy}, 500, 1,
                -1 /* uniform initial state distribution */, false);

    SoftmaxDirichletPrior prior(actions);
    DirichletPolicyPosterior foo(prior, demonstrations);

    vector<int>& r5 = foo.getActionCounts(5);
    r5[5] = 555;
    vector<int>& r2 = foo.getActionCounts(2);
    r2[2] = 222;

    cout << "r5:\t";
    for (int c : r5)
        cout << c << " ";
    cout << endl;
    cout << "r2:\t";
    for (int c : r2)
        cout << c << " ";
    cout << endl;
    cout << "r2 ?:\t";
    for (int c : r2)
        cout << c << " ";
    cout << endl;

    vector<int> tests = {2,5,0};
    for (int test : tests)
    {
        for (int p = 0; p < 2; ++p)
        {
            Policy& pi = foo.samplePolicy();
            vector<pair<int,double>> ssmn;
            for (int i = 0; i < 2; ++i)
            {
                ssmn = pi.probabilities(test);
                cout << "\ts" << test << "mn(" << i << ") policy " << p << "\t";
                for (auto c : ssmn)
                    cout << c.second << " ";
                cout << endl;
            }
        }
    }

    return;
}
