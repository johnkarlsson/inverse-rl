#ifndef BMT_H
#define BMT_H

#include <vector>
#include <set>

#include "../model/DiscreteCMP.h"
#include "../model/Policy.h"
#include "../model/random_mdp/RandomMDP.h"
#include "../algorithm/LSTDQ.h" // TODO: For Demonstration

using std::vector;
using std::set;

class BMT
{
    public:
        BMT( FeatureMDP mdp, vector<Demonstration>& D,
             vector<vector<double>> const & rewardFunctions,
             vector<DeterministicPolicy> const & optimalPolicies,
             vector<Policy*> const & policies,
             double _c, bool withModel = true,
             bool sum = false);
        double loss(vector<double> const & weightsEval,
                    vector<double> const & weightsOpt,
                    bool calculateSum = false);
        static double loss(vector<double> const & weightsEval,
                           vector<double> const & weightsOpt,
                           set<int> const & states, DiscreteCMP const & cmp,
                           bool sum = false);
        double getLoss(int policy, int rewardFunction);

        double getRewardProbability(int rewardFunction);
        vector<double> getNormalizedRewardProbabilities();

        DeterministicPolicy optimalPolicy();

        // Linalg
        static vector<double> solve_rect(vector<double> A, vector<double> b);

        double beta(double ep_lower, double ep_upper);

    private:
        const int K;
        const int N;
        vector<vector<double>> policyRewardLoss;
        vector<double> sortedLosses;
        // set<double> sortedLosses;
        FeatureMDP mdp;
        vector<vector<double>> const & rewardFunctions;
        vector<Demonstration> const & lstdqDemonstrations;
        set<int> states;
        const double c;
};

#endif
