#ifndef TICTACTOEMDP_H
#define TICTACTOEMDP_H

#include "../DiscreteMDP.h"

class TicTacToeMDP
    : public FeatureMDP
{
    public:
        TicTacToeMDP(const DiscreteCMP * cmp, bool _trueRewards);

        double getReward(int s) const;
        // void setReward(int s, double r);
        // void setRewardWeights(std::vector<double> weights);

    private:
        // std::vector<double> rewardWeights;
        bool trueRewards;
};

#endif
