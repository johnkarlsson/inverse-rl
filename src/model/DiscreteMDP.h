#ifndef DISCRETEMDP_H
#define DISCRETEMDP_H

#include "DiscreteCMP.h"
#include <vector>
#include <numeric>
#include <stdexcept>

class DiscreteMDP
{
    public:
        DiscreteMDP(const DiscreteCMP * cmp, double gamma,
                    bool initRewardVector = true);

        virtual double getReward(int s) const;
        virtual void setReward(int s, double r);

        const DiscreteCMP * const cmp;
        double gamma;

    private:
        std::vector<double> rewards; // direct rewards for states
};

class FeatureMDP
    : public DiscreteMDP
{
    public:
        FeatureMDP(const DiscreteCMP * cmp, double gamma)
            : DiscreteMDP(cmp, gamma, false), rewardWeights(cmp->nFeatures())
        {}
        void setRewardWeights(std::vector<double> weights)
            { rewardWeights = weights; }
        std::vector<double> getRewardWeights()
            { return rewardWeights; }
        double getReward(int s) const
        {
            auto phi = cmp->features(s);
            return inner_product(phi.begin(), phi.end(), rewardWeights.begin(),
                                 0.0);
        }
        void setReward(int s, double r)
        {
            throw std::runtime_error(
                    "Undefined function FeatureMDP::setReward");
        }

    private:
        std::vector<double> rewardWeights;
};

#endif
