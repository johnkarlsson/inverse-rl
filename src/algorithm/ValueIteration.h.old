#ifndef VALUEITERATION_H
#define VALUEITERATION_H

#include "../model/DiscreteMDP.h"
#include <vector>

class ValueIteration
{
    public:
        ValueIteration(const DiscreteMDP * const mdp);
        void computeStateActionValues(double epsilon = 0.001);
        void init(double epsilon = 0.001);
        void computeStateValues(double epsilon = 0.001);
        std::vector<std::vector<double>> Q;
        std::vector<double> V;
    private:
        const DiscreteMDP * const mdp;
};

#endif
