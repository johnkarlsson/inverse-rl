#ifndef BMT_H
#define BMT_H

#include <vector>

#include "../model/DiscreteCMP.h"
#include "../model/Policy.h"
#include "../model/random_mdp/RandomMDP.h"
#include "../algorithm/LSTDQ.h" // TODO: For Demonstration

using std::vector;

class BMT
{
    public:
        BMT( RandomMDP mdp, vector<Demonstration> D,
             vector<vector<double>*> const & rewardFunctions,
             vector<Policy*> const & policies);
        static double loss(vector<double> const & weightsEval, // TODO: Make private
                           vector<double> const & weightsOpt,
                           vector<int> const & states, DiscreteCMP const & cmp);
        double getLoss(int policy, int rewardFunction);

        double getRewardProbability(int rewardFunction);

    private:
        const int K;
        const int N;
        vector<vector<double>> policyRewardLoss;
        vector<double> sortedLosses;
};

#endif
