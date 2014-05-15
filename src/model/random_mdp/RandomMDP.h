#ifndef RANDOMMDP_H
#define RANDOMMDP_H

#include "RandomCMP.h"
#include "../DiscreteMDP.h"

#include <vector>

class RandomMDP
    : public DiscreteMDP
{
    public:
        RandomMDP(const RandomCMP * cmp, double gamma);

        double getReward(int s) const;
        void setReward(int s, double r); // TODO: Skip and create new class
        void setRewardWeights(std::vector<double> weights);
        std::vector<double> getRewardWeights();

    private:
        std::vector<double> rewardWeights;
};

#endif
