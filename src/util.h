#ifndef UTIL_H
#define UTIL_H

#include "model/Policy.h"
#include "model/Transition.h"
#include "model/DiscreteMDP.h"

using namespace std;

double getAverageOptimalUtility(Policy& optimalPolicy, DiscreteMDP& mdp,
                                int state, int nPlayouts, int T);

double getExpectedOptimalReward(Policy& optimalPolicy, DiscreteMDP& mdp,
                                int state, int action, int nPlayouts, int T);

vector<Demonstration> generateDemonstrations(DiscreteMDP& mdp,
                                             vector<Policy*> policies,
                                             int demonstrationLength,
                                             int nDemonstrations = 1,
                                             int initialState = -1,
                                             bool print = true);

vector<vector<double>> sampleRewardFunctions(int nFunctions, int nPlayouts,
                                             int playoutHorizon,
                                             vector<Demonstration>&
                                                 lstdqDemonstrations,
                                             Policy& optimalPolicy,
                                             DiscreteMDP& mdp);

int sample(vector<pair<int, double>> transitionProbabilities);

inline double r() { return ((double) rand() / (double) RAND_MAX); }

void normalize(vector<double>& v);

#endif
