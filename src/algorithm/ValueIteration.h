#ifndef VALUEITERATION_H
#define VALUEITERATION_H

#include "../model/DiscreteMDP.h"
#include <vector>

class ValueIteration
{
    public:
        ValueIteration(const DiscreteMDP * const mdp);
        void computeStateActionValues(double epsilon = 1e-6);
        std::vector<std::vector<double>> Q;
    private:
        const DiscreteMDP * const mdp;
};

#endif
