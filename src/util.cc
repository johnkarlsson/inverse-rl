#include "util.h"
#include "model/Policy.h"
#include "model/Transition.h"
#include "model/DiscreteMDP.h"

#include <sstream>

vector<Demonstration> generateDemonstrations(DiscreteMDP& mdp,
                                             vector<Policy*> policies,
                                             int demonstrationLength,
                                             int nDemonstrations,
                                             int initialState,
                                             bool print)
{
    if (print)
    {
        cout << "Generating " << nDemonstrations << " demonstrations";
        if (demonstrationLength > -1)
            cout << " of length " << demonstrationLength;
        cout << "..." << endl;
    }

    vector<Demonstration> demonstrations(nDemonstrations);

    for (int d = 0; d < nDemonstrations; ++d)
    {
        int currentState;
        if (initialState == -1) // Uniform initial state dist
            currentState = rand() % mdp.cmp->states;
        else
            currentState = initialState;
        int previousState;

        for (int t = 0;
             (demonstrationLength < 0 || t < demonstrationLength)
                 && !mdp.cmp->isTerminal(currentState);
             ++t)
        {
            Policy& pi = *policies[t % policies.size()];
            auto actionProbabilities = pi.probabilities(currentState);
            int action = sample(actionProbabilities);

            previousState = currentState;
            auto transitionProbabilities = 
                mdp.cmp->kernel->getTransitionProbabilities(currentState,
                                                            action);
            currentState = sample(transitionProbabilities);

            double reward = mdp.getReward(currentState);
            demonstrations[d].push_back(Transition(previousState, action,
                                                   currentState, reward));
        }
    }

    if (print)
        cout << "Done!" << endl;
    return demonstrations;
}

int sample(vector<pair<int, double>> transitionProbabilities)
{
    double rr = r(); // [0,1]
    double sum = 0;
    for (auto p : transitionProbabilities)
    {
        int s2 = p.first;
        double prob = p.second;
        sum += prob;
        if (rr <= sum) // Assumes that the probabilities sum to 1
            return s2;
    }
    if (sum < 1-1e-6)
    {
        std::ostringstream ss;
        ss << "sum = " << sum << " < 1";
        throw std::runtime_error(ss.str());
    }
    else
    {
        return transitionProbabilities.back().first;
    }
}

double getAverageOptimalUtility(Policy& optimalPolicy, DiscreteMDP& mdp,
                                int state, int nPlayouts, int T)
{
    vector<Demonstration> playouts
        = generateDemonstrations(mdp, {&optimalPolicy}, T, nPlayouts, state,
                                 false);
    // Calculate discounted sum of rewards
    double totalPayoff = 0;
    for (Demonstration d : playouts)
        for (int t = 0; t < d.size(); ++t)
            totalPayoff += d[t].r * pow(mdp.gamma, t); // starts with ^0
    return (totalPayoff / (double) nPlayouts);
}

void normalize(vector<double>& v)
{
    double sum = 0;
    int k = v.size();
    for (int i = 0; i < k; ++i)
        sum += v[i];
    for (int i = 0; i < k; ++i)
        v[i] /= sum;
}

