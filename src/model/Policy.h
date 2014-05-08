#ifndef POLICY_H
#define POLICY_H

#include <vector>
#include <numeric>
#include <cfloat>
#include <iostream>
#include <assert.h>
#include "DiscreteCMP.h"

using std::vector;

class Policy
{
    public:
        virtual std::vector<std::pair<int,double>> probabilities(int s) = 0;
        virtual ~Policy()
        {
            std::cout << "~Policy()" << std::endl;
        };
};

class ConstPolicy 
    : public Policy
{
    public:
        ConstPolicy(int _action, int states) : actions(states, _action) {}
        ConstPolicy(vector<int> _actions) : actions(_actions) {}
        std::vector<std::pair<int,double>> probabilities(int s)
        { return {{actions[s],1}}; }
    private:
        vector<int> actions;
};

class DeterministicPolicy
    : public Policy
{
    public:
        DeterministicPolicy(DiscreteCMP const * const _cmp,
                            vector<double> _weights)
            : cmp(_cmp), weights(_weights)
        {}
        DeterministicPolicy(DiscreteCMP const * const _cmp)
            : cmp(_cmp), weights(_cmp->nFeatures(), 0)
        {}

        const vector<double>& getWeights() const
        {
            return weights;
        }

        void setWeights(vector<double> w)
        {
            assert(w.size() == weights.size());
            std::copy(w.begin(), w.end(), weights.begin());
        }

        std::vector<std::pair<int,double>> probabilities(int s)
        {
            auto validActions = cmp->kernel->getValidActions(s);
            int aMax;
            double qMax = -DBL_MAX;
            for (int a : validActions)
            {
                auto phi = cmp->features(s,a);
                double q = std::inner_product(phi.begin(), phi.end(),
                                              weights.begin(), 0.0);
                if (q > qMax)
                {
                    qMax = q;
                    aMax = a;
                }
            }
            return {{aMax,1}};
        }

    private:
        DiscreteCMP const * const cmp;
        vector<double> weights;
};

#endif
