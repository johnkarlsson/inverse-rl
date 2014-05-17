#ifndef RANDOMTRANSITIONKERNEL_H
#define RANDOMTRANSITIONKERNEL_H

#include "../TabularTransitionKernel.h"
#include <vector>
#include <gsl/gsl_rng.h>

class RandomTransitionKernel
    : public TabularTransitionKernel
{
    public:
        RandomTransitionKernel(int _states, int _actions);
        void test_dirichlet();
        std::vector<double> sample_multinomial();
    private:
        // const double alpha = 0.25;
        const double alpha = 1;
        const std::vector<double> alphas;
        gsl_rng *r_global;
};

#endif
