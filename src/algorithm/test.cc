#include "ValueIteration.h"
#include "LSPI.h"

#include "../model/Policy.h"
#include "../model/random_mdp/RandomMDP.h"

#include "../algorithm/BMT.h"

#include "../util.h"

vector<double> test_lstdq_randomMDP(RandomMDP& mdp, Policy& pi)
{
    RandomPolicy randomPolicy(mdp.cmp);
    auto demonstrations =
        generateDemonstrations(mdp, {&randomPolicy}, 500, 1,
                -1 /* uniform initial state distribution */, false);
    const int k = mdp.cmp->nFeatures();

    vector<double> w = LSPI::lstdq(demonstrations, pi, mdp);
    cout << "LSTDQ w* ( ";
    for (int s = 0; s < mdp.cmp->states; ++s)
    {
        cout << pi.probabilities(s)[0].first << " ";
    }
    cout << ") = ";
    for (int i = 0; i < k; ++i)
        cout << w[i] << " ";
    cout << endl;

    return w;
}

void test_valueiteration()
{
    srand((unsigned)time(NULL));
    cout << "*** Testing ValueIteration on RandomMDP" << endl;

    const double gamma = 0.5;
    const int states = 3;
    const int actions = 2;
    RandomTransitionKernel kernel(states, actions); // All values overridden
                                                    // below
    RandomCMP cmp(&kernel);
    RandomMDP mdp(&cmp, gamma);

    kernel.setTransitionProbability(0, 0, 0, 0.5);
    kernel.setTransitionProbability(0, 0, 1, 0.5);
    kernel.setTransitionProbability(0, 0, 2, 0.0);

    kernel.setTransitionProbability(1, 0, 0, 0.0);
    kernel.setTransitionProbability(1, 0, 1, 1.0);
    kernel.setTransitionProbability(1, 0, 2, 0.0);

    kernel.setTransitionProbability(2, 0, 0, 1.0);
    kernel.setTransitionProbability(2, 0, 1, 0.0);
    kernel.setTransitionProbability(2, 0, 2, 0.0);


    kernel.setTransitionProbability(0, 1, 0, 1.0);
    kernel.setTransitionProbability(0, 1, 1, 0.0);
    kernel.setTransitionProbability(0, 1, 2, 0.0);

    kernel.setTransitionProbability(1, 1, 0, 0.5);
    kernel.setTransitionProbability(1, 1, 1, 0.0);
    kernel.setTransitionProbability(1, 1, 2, 0.5);

    kernel.setTransitionProbability(2, 1, 0, 0.5);
    kernel.setTransitionProbability(2, 1, 1, 0.0);
    kernel.setTransitionProbability(2, 1, 2, 0.5);

    mdp.setRewardWeights({-5, 1, 10});

    ValueIteration vi(&mdp);
    vi.computeStateActionValues();

    cout.precision(numeric_limits< double >::digits10 - 12);
    for (int a = 0; a < actions; ++a)
    {
        cout << "A" << a << ":\t";
        for (int s = 0; s < states; ++s)
            cout << fixed << vi.Q[s][a] << " | \t";
        cout << endl;
    }

    cout << "V" << ":\t";
    for (int s = 0; s < states; ++s)
    {
        double qMax = -DBL_MAX;
        int aMax;
        for (int a = 0; a < actions; ++a)
        {
            double q = vi.Q[s][a];
            if (q > qMax)
            {
                qMax = q;
                aMax = a;
            }
        }
        cout << fixed << qMax << "/A" << aMax << "  \t";
    }
    cout << endl;

    cout << "R" << ":\t";
    for (int s = 0; s < states; ++s)
        cout << fixed << mdp.getReward(s) << " | \t";
    cout << endl;

    // Print transition probs for each state and action
    cout << "Transition probabilities from states given actions:" << endl;
    for (int s = 0; s < states; ++s)
    {
        cout << "\tS" << s << ":" << endl;
        for (int a = 0; a < actions; ++a)
        {
            cout << "\t\tA" << a << ":\t";
            for (int s2 = 0; s2 < states; ++s2)
            {
                cout << kernel.getTransitionProbability(s, a, s2) << "\t";
            }
            cout << endl;
        }
    }

    vector<int> bestActions(states, 0);
    for (int s = 0; s < states; ++s)
    {
        int aMax;
        double qMax=-DBL_MAX;
        for (int a = 0; a < actions; ++a)
        {
            double q = vi.Q[s][a];
            if (q > qMax)
            {
                qMax = q;
                aMax = a;
            }
        }
        bestActions[s] = aMax;
    }

    ConstPolicy pi00(0, states);
    ConstPolicy pi1(1, states);
    ConstPolicy piOpt(bestActions);

    test_lstdq_randomMDP(mdp, pi00);
    test_lstdq_randomMDP(mdp, pi1);
    vector<double> w = test_lstdq_randomMDP(mdp, piOpt);

    /*
     * LSPI
     */
    RandomPolicy randomPolicy(mdp.cmp);
    auto demonstrations =
        generateDemonstrations(mdp, {&randomPolicy}, 500, 1,
                -1 /* uniform initial state distribution */, false);

    DeterministicPolicy lspiPolicy = LSPI::lspi(demonstrations, mdp);
    cout << "LSPI w* ( ";
    for (int s = 0; s < mdp.cmp->states; ++s)
    {
        cout << lspiPolicy.probabilities(s)[0].first << " ";
    }
    cout << ") = ";
    for (int i = 0; i < mdp.cmp->nFeatures(); ++i)
        cout << lspiPolicy.getWeights()[i] << " ";
    cout << endl;
    /*
     */

    cout << "{V}:\t";
    for (int s = 0; s < states; ++s)
    {
        double avgValue = getAverageOptimalUtility(lspiPolicy, mdp, s, 100,
                                                   100 /* -1 for ttt */);
        cout << fixed << avgValue << "  \t\t";
    }
    cout << endl;

    cout << "~Q* - Q*" << endl;
    cout << "S\\A\t";
    for (int a = 0; a < actions; ++a)
        cout << a << "\t";
    cout << endl;
    for (int s = 0; s < states; ++s)
    {
        cout << s << "\t";
        for (int a = 0; a < actions; ++a)
        {
            vector<double> phi = cmp.features(s, a);
            double q = std::inner_product(phi.begin(), phi.end(), w.begin(), 0.0);
            cout << q - vi.Q[s][a] << "\t";
        }
        cout << endl;
    }

    int n = states * actions;
    int k = cmp.nFeatures();
    vector<double> A(n*k);
    vector<double> b(n);
    int i = 0;
    int j = 0;
    for (int s = 0; s < states; ++s)
    {
        for (int a = 0; a < actions; ++a)
        {
            auto tp = cmp.kernel->getTransitionProbabilities(s, a);
            double avgValue = 0;
            double avgReward = 0;
            for (auto tr : tp)
            {
                int s2 = tr.first;
                double p = tr.second;
                double v = -DBL_MAX;
                for (int a2 = 0; a2 < actions; ++a2)
                {
                    double q = vi.Q[s2][a2];
                    if (q > v)
                        v = q;
                }
                avgValue +=  p * v;
                avgReward += p * mdp.getReward(s2);
            }
            double approxAvgReward = vi.Q[s][a] - gamma * avgValue;
            double sampledAvgReward = getExpectedOptimalReward(lspiPolicy,
                                                               mdp, s, a,
                                                               5, 100);
            cout << "\tApproximate avg reward\t~<Q(" << s << "," << a << ")> =\t"
                 << approxAvgReward << endl;
            cout << "\tTrue avg reward\t\t <Q(" << s << "," << a << ")> =\t" << avgReward << endl;
            cout << "\tSampled avg reward\t {Q(" << s << "," << a << ")} =\t"
                 << sampledAvgReward << endl;

            vector<double> phi = cmp.features(s,a);
            for (double f : phi)
                A[i++] = f;
            // b[j++] = approxAvgReward;
            b[j++] = sampledAvgReward;
        }
    }

    // auto rewards = BMT::solve_rect(A, b);

    cout << "Solved rewards from linear system generated using features &" 
         << " sampled average rewards:" << endl;
    auto rewardFunctions = sampleRewardFunctions(5, 1, 50, demonstrations,
                                                 lspiPolicy, mdp);
    for (auto rewardFunction : rewardFunctions)
    {
        for (auto r : rewardFunction)
            cout << r << "\t";
        cout << endl;
    }

    vector<double> phi = cmp.features(2, 0);
    cout << "phi(s=2, a=0) = ";
    for (auto f : phi)
    {
        cout << f << " ";
    }
    cout << endl;
    auto tp = cmp.kernel->getTransitionProbabilities(2, 0);
    for (auto p : tp)
    {
        cout << "P(s'=" << p.first << " | s=2, a=0) = " << p.second << "\t";
    }
    cout << endl;



    ConstPolicy pi0(0, states);
    vector<double> w0 = test_lstdq_randomMDP(mdp, pi0);

    set<int> demoStates;
    for (Demonstration demo : demonstrations)
        for (Transition tr : demo)
            demoStates.insert(tr.s);

    double loss = BMT::loss(w0, lspiPolicy.getWeights(), demoStates, cmp);
    cout << "Loss between pi(.)=a0 and pi_lspi(.) compared over "
         << demoStates.size() << " states is: " << loss << endl;
}
