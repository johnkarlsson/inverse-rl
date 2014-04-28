#include "TabularTransitionKernel.h"
#include <vector>
#include <set>
#include <algorithm>

using namespace std;

TabularTransitionKernel::TabularTransitionKernel(int _states, int _actions)
    : TransitionKernel(_states, _actions),
      valid_actions(states, set<int>()),
      kernel(states, vector<vector<double>>(actions, vector<double>(states)))
{
    // kernel = vector<vector<vector<double>>>(
    //         states, vector<vector<double>>(
    //             actions, vector<double>(
    //                 states)));
}

TabularTransitionKernel::~TabularTransitionKernel()
{}

double TabularTransitionKernel::getTransitionProbability(const int s,
                                             const int a,
                                             const int s2) const
{
    return kernel[s][a][s2];
}

void TabularTransitionKernel::setTransitionProbability(const int s,
                                             const int a,
                                             const int s2,
                                             const double p)
{
    if (p != 0)
        valid_actions[s].insert(a);
    else
    {
        bool nonzero = false;
        for (int ss2 = 0; ss2 < states; ++ss2)
            if (kernel[s][a][ss2] != 0)
            {
                nonzero = true;
                break;
            }
        if (nonzero)
            valid_actions[s].erase(a);
    }

    kernel[s][a][s2] = p;
}

vector< pair<state, probability> >
    TabularTransitionKernel::getTransitionProbabilities(const int s,
                                                        const int a) const
{
    vector< pair<state, probability> > output;
    for (int s2 = 0; s2 < states; ++s2)
    {
        double p = kernel[s][a][s2];
        if (p != 0)
            output.push_back({s2, p});
    }

    return output;
}

set<int> TabularTransitionKernel::getValidActions(const int s) const
{
    return valid_actions[s];
}
