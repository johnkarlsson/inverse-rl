#include "BMT.h"

#include <vector>
#include <cfloat>
#include <cmath>
#include <numeric>
#include <algorithm>

#include <gsl/gsl_cdf.h>

#include "../model/DiscreteCMP.h"

using std::vector;
using std::inner_product;



BMT::BMT( FeatureMDP _mdp,
          vector<Demonstration>& _lstdqDemonstrations,
          vector<vector<double>> const & _rewardFunctions,
          vector<DeterministicPolicy> const & optimalPolicies,
          vector<Policy*> const & sampledPolicies,
          double _c, bool withModel, bool sum,
          set<int>* _states)
    : K(sampledPolicies.size()), N(_rewardFunctions.size()),
      mdp(_mdp), rewardFunctions(_rewardFunctions),
      lstdqDemonstrations(_lstdqDemonstrations),
      c(_c)
{
    if (_states != NULL)
        states = set<int>(*_states);
    else
        for (Demonstration demo : lstdqDemonstrations)
            for (Transition tr : demo)
                states.insert(tr.s);

    policyRewardLoss = vector<vector<double>>(K, vector<double>(N, 0));
    for (int j = 0; j < N; ++j)
    {
        mdp.setRewardWeights(rewardFunctions[j]);
        // vector<double> weightsOpt = LSTDQ::lspi(D, mdp).getWeights();
        vector<double> weightsOpt = optimalPolicies[j].getWeights();
        for (int k = 0; k < K; ++k)
        {
            vector<double> weightsEval = LSTDQ::lstdq(lstdqDemonstrations,
                                                      *sampledPolicies[k], mdp,
                                                      withModel);
            policyRewardLoss[k][j] = loss(weightsEval, weightsOpt,
                                          states, *mdp.cmp, sum);
        }
    }

    // for (auto v : policyRewardLoss)
    //     for (auto loss : v)
    //         sortedLosses.insert(loss);

    for (auto v : policyRewardLoss)
        for (auto loss : v)
            sortedLosses.push_back(loss);
    std::sort(sortedLosses.begin(), sortedLosses.end()); // Ascending
}

vector<double> BMT::getNormalizedRewardProbabilities()
{
    vector<double> output;
    double sum = 0;
    for (int i = 0; i < N; ++i)
    {
        double p = getRewardProbability(i);
        sum += p;
        output.push_back(p);
    }
    for (int i = 0; i < N; ++i)
        output[i] /= sum;
    return output;
}

double BMT::loss(vector<double> const & weightsEval,
                 vector<double> const & weightsOpt,
                 bool calculateSum)
{
    return BMT::loss(weightsEval, weightsOpt, states, *mdp.cmp, calculateSum);
}

double BMT::loss(vector<double> const & wEval,
                 vector<double> const & wOpt,
                 set<int> const & states, DiscreteCMP const & cmp,
                 bool calculateSum)
{
    double sup = -DBL_MAX;
    // TODO: Do this for all state action pairs in a demonstration instead?
    // Probably not though
    double sum = 0;
    for (int s : states)
    {
        auto validActions = cmp.kernel->getValidActions(s);
        double vOpt = -DBL_MAX;
        double vEval = -DBL_MAX;
        for (auto a : validActions) // V(s) = max_a Q(s,a)
        {
            vector<double> phi = cmp.features(s,a);
            double vOpt_tmp = inner_product(phi.begin(), phi.end(), wEval.begin(), 0.0);
            double vEval_tmp = inner_product(phi.begin(), phi.end(), wOpt.begin(), 0.0);
            if (vOpt < vOpt_tmp)
                vOpt = vOpt_tmp;
            if (vEval < vEval_tmp)
                vEval = vEval_tmp;
        }
        // vector<double> phi = cmp.features(s);
        // double vOpt = inner_product(phi.begin(), phi.end(), wEval.begin(), 0.0);
        // double vEval = inner_product(phi.begin(), phi.end(), wOpt.begin(), 0.0);

        double diff = fabs(vOpt - vEval);
        if (calculateSum)
            sum += diff * diff;
        else
            if (diff > sup)
                sup = diff;
    }

    if (calculateSum)
        return sqrt(sum / (double) states.size());
    else
        return sup;
}

double BMT::getLoss(int policy, int rewardFunction)
{
    return policyRewardLoss[policy][rewardFunction];
}

double BMT::beta(double ep_lower, double ep_upper)
{
    // const static double c = 0.1;
    //return (gsl_cdf_gamma_P(ep_upper, c, 0.1)
    //        - gsl_cdf_gamma_P(ep_lower, c, 0.1));
    return (exp(-c*ep_lower) - exp(-c*ep_upper));
}

double BMT::getRewardProbability(int rewardFunction) // Psi(B | ep, pi)
{
    const int j = rewardFunction;
    double probabilitySum = 0.0;

    for (int k = 0; k < K; ++k)
    {
        for (int i = 0; i < sortedLosses.size(); ++i)
        {
            double ep = sortedLosses[i];
            double ep_upper = (i == sortedLosses.size() - 1)
                                ? DBL_MAX : sortedLosses[i+1];
            // a rho that is ep-optimal is also ep_upper-optimal

            if (getLoss(k, j) <= ep)
            {
                int nEpOptimalRewardFunctions = 0;
                for (int l = 0; l < N; ++l)
                    if (getLoss(k,l) <= ep)
                        ++nEpOptimalRewardFunctions;
                probabilitySum += beta(ep, ep_upper)
                                    / (double) nEpOptimalRewardFunctions;
            }
        }
    }

    return probabilitySum;
}

// Ax = b, returns x for rectangular systems
vector<double> BMT::solve_rect(vector<double> A, vector<double> b)
{
    int n = b.size();
    int k = A.size() / n;

    assert(A.size() == n*k);
    assert(n > k && "Matrix A in rectangular system Ax=b must have more rows than columns");

    gsl_matrix_view _A = gsl_matrix_view_array(A.data(), n, k);
    gsl_vector_view _b = gsl_vector_view_array(b.data(), n);

    gsl_vector *x = gsl_vector_alloc (k);
    gsl_vector *tau = gsl_vector_alloc(k);
    gsl_vector *gsl_residual = gsl_vector_alloc(n);

    gsl_linalg_QR_decomp (&_A.matrix, tau);
    gsl_linalg_QR_lssolve (&_A.matrix, tau, &_b.vector, x, gsl_residual);

    vector<double> output(x->data, x->data + k);

    gsl_vector_free (x);
    gsl_vector_free (tau);
    gsl_vector_free (gsl_residual);

    return output;
}

// double BMT::optimalPolicyLoss()
// {
//     Policy& pi = optimalPolicy();
// }

DeterministicPolicy BMT::optimalPolicy()
{
    int maxRewardFunction = 0;
    double maxProbability = -DBL_MAX;
    for (int i = 0; i < N; ++i)
    {
        double p = getRewardProbability(i);
        if (p > maxProbability)
        {
            maxRewardFunction = i;
            maxProbability = p;
        }
    }

    mdp.setRewardWeights(rewardFunctions[maxRewardFunction]);

    return LSTDQ::lspi(lstdqDemonstrations, mdp);
}

