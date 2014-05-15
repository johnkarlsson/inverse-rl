#ifndef POLICY_H
#define POLICY_H

#include <vector>
#include <numeric>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <assert.h>
#include "DiscreteCMP.h"

using std::vector;

class Policy
{
    public:
        virtual std::vector<std::pair<int,double>> probabilities(int s) = 0;
        virtual ~Policy()
        {
            // std::cout << "~Policy()" << std::endl;
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

class FeaturePolicy // Abstract
    : public Policy
{
    public:
        FeaturePolicy(DiscreteCMP const * const _cmp,
                            vector<double> _weights)
            : weights(_weights), cmp(_cmp)
        {}
        FeaturePolicy(DiscreteCMP const * const _cmp)
            : weights(_cmp->nFeatures(), 0), cmp(_cmp)
        {}


        double getValue(int s, int a)
        {
            auto phi = cmp->features(s,a);
            double q = std::inner_product(phi.begin(), phi.end(),
                                          weights.begin(), 0.0);
            return q;
        }

        const vector<double>& getWeights() const
        {
            return weights;
        }

        void setWeights(vector<double> w)
        {
            assert(w.size() == weights.size());
            std::copy(w.begin(), w.end(), weights.begin());
        }

    protected:
        vector<double> weights;
        DiscreteCMP const * const cmp;
};

// class SoftmaxPolicy
class DeterministicPolicy
    : public FeaturePolicy
{
    public:
        DeterministicPolicy(DiscreteCMP const * const _cmp,
                            vector<double> _weights)
            : FeaturePolicy(_cmp, _weights)
        {}
        DeterministicPolicy(DiscreteCMP const * const _cmp)
            : FeaturePolicy(_cmp, vector<double>(_cmp->nFeatures(), 0))
        {}

        std::vector<std::pair<int,double>> probabilities(int s)
        {
            auto validActions = cmp->kernel->getValidActions(s);
            int aMax;
            double qMax = -DBL_MAX;
            for (int a : validActions)
            {
                double q = getValue(s,a);
                if (q > qMax)
                {
                    qMax = q;
                    aMax = a;
                }
            }
            return {{aMax,1}};
        }
};

class SoftmaxPolicy
    : public FeaturePolicy
{
    public:
        SoftmaxPolicy(DiscreteCMP const * const _cmp,
                            vector<double> _weights, double _temperature)
            : FeaturePolicy(_cmp, _weights), c(_temperature)
        {}
        SoftmaxPolicy(DiscreteCMP const * const _cmp, double _temperature)
            : FeaturePolicy(_cmp, vector<double>(_cmp->nFeatures(), 0)),
              c(_temperature)
        {}

        std::vector<std::pair<int,double>> probabilities(int s)
        {
            std::set<int> validActions = cmp->kernel->getValidActions(s);
            std::vector<std::pair<int,double>> output(validActions.size());

            std::vector<double> qValues(validActions.size());
            int i = 0;
            for (int a : validActions)
                qValues[i++] = getValue(s,a);
            double qMax = *std::max_element(qValues.begin(), qValues.end());
            for (int i = 0; i < qValues.size(); ++i)
                qValues[i] -= qMax;
            double sum = 0;
            i = 0;
            for (int a : validActions)
            {
                double q = qValues[i];
                double e = exp(q / c);
                output[i] = {a, e};
                sum += e;
                ++i;
            }
            for (int i = 0; i < output.size(); ++i)
                output[i].second /= sum;
            return output;
        }
    private:
        const double c;
};

#endif
