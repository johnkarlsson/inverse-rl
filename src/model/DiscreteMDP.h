#ifndef DISCRETEMDP_H
#define DISCRETEMDP_H

#include "DiscreteCMP.h"
#include <vector>

class DiscreteMDP
{
    public:
    DiscreteMDP(const DiscreteCMP *cmp, double gamma);

    double getReward(int s) const;
    void setReward(int s, double r);

    const DiscreteCMP * const cmp;
    const double gamma;

    private:
    std::vector<double> rewards; // direct rewards for states
};


#endif
