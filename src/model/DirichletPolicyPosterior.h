#ifndef DIRICHLETPOLICYPOSTERIOR_H
#define DIRICHLETPOLICYPOSTERIOR_H

#include "Policy.h"
#include "../algorithm/LSTDQ.h" // TODO: Just for Demonstration and Transition, move them!
#include <vector>
#include <iostream>
#include <memory>
#include <map>
#include <gsl/gsl_rng.h>

class StateActionDirichlet
{
    public:
        StateActionDirichlet(int _actions) : actions(_actions)
        {
            if (!r_global)
            {
                r_global = gsl_rng_alloc(gsl_rng_default);
                gsl_rng_set(r_global, (unsigned)time(NULL));
            }
        }
        virtual double getAlpha(int s, int a) = 0;
        void sample(int s, double * output);
        const int actions;
    private:
        static gsl_rng * r_global;
};

class SoftmaxDirichletPrior
    : public StateActionDirichlet
{
    public:
        SoftmaxDirichletPrior(int _actions) : StateActionDirichlet(_actions) {}
        double getAlpha(int s, int a) { return 1; }
};

class DirichletPolicyPosterior
    : StateActionDirichlet
{
    public:
        DirichletPolicyPosterior(StateActionDirichlet& _prior,
                                 vector<Demonstration> const & D)
            : StateActionDirichlet(_prior.actions), prior(_prior)
        {
            for (Demonstration d : D)
                for (Transition tr : d)
                    getActionCounts(tr.s)[tr.a]++;
        }

        Policy& samplePolicy();
        double getAlpha(int s, int a);
        std::vector<int>& getActionCounts(int s);

    private:
        DirichletPolicyPosterior(int _actions);
        DirichletPolicyPosterior();
        StateActionDirichlet& prior;
        std::map<int, vector<int>> stateActionCounts;
        class SampledPolicy
            : public Policy
        {
            public:
                SampledPolicy(DirichletPolicyPosterior * const _parent)
                    : parent(_parent) {}
                std::vector<std::pair<int,double>> probabilities(int s);
            private:
                DirichletPolicyPosterior * const parent;
                std::map<int, vector<double>> stateMultinomials;
                vector<double>& getMultinomial(int s);
        };
        static std::vector<std::unique_ptr<Policy>> sampledPolicies;
};

#endif
