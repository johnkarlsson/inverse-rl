#ifndef DIRICHLETPOLICYPOSTERIOR_H
#define DIRICHLETPOLICYPOSTERIOR_H

#include "Policy.h"
#include <vector>
#include <iostream>
#include <memory>
#include <map>
#include <gsl/gsl_rng.h>

class DirichletPolicyPosterior
{
    public:
        DirichletPolicyPosterior()
            : actions(10)
        {
            if (!r_global)
            {
                r_global = gsl_rng_alloc(gsl_rng_default);
                gsl_rng_set(r_global, (unsigned)time(NULL));
            }
        }
        Policy& samplePolicy();
        ~DirichletPolicyPosterior()
        {
            std::cout << "~DirichletPolicyPosterior()" << std::endl;
        }

        std::vector<int>& getStateActionCounts(int s);

    private:
        const int actions;
        std::map<int, vector<int>> stateActionCounts;
        static gsl_rng * r_global;
        class SampledPolicy
            : public Policy
        {
            public:
                std::vector<std::pair<int,double>> probabilities(int s) const;
        };
        static std::vector<std::unique_ptr<Policy>> sampledPolicies;
};

#endif
