#include "BMT.h"

#include <vector>
#include <cfloat>
#include <cmath>
#include <numeric>
#include <algorithm>

#include "../model/DiscreteCMP.h"

using std::vector;
using std::inner_product;



BMT::BMT( RandomMDP mdp, vector<Demonstration> D,
          vector<vector<double>*> const & rewardFunctions,
          vector<Policy*> const & policies)
    : K(policies.size()), N(rewardFunctions.size())
{
    vector<int> states(D.size());
    for (Demonstration demo : D)
        for (Transition tr : demo)
            states.push_back(tr.s); // TODO: Alot of repetition of states here. Make it std::set
    policyRewardLoss = vector<vector<double>>(K, vector<double>(N, 0));
    for (int j = 0; j < N; ++j)
    {
        mdp.setRewardWeights(*rewardFunctions[j]);
        vector<double> weightsOpt = LSTDQ::lspi(D, mdp).getWeights();
        for (int k = 0; k < K; ++k)
        {
            vector<double> weightsEval = LSTDQ::lstdq(D, *policies[k],
                                                      mdp);

            policyRewardLoss[k][j] = loss(weightsEval, weightsOpt,
                                          states, *mdp.cmp);
        }
    }

    for (auto v : policyRewardLoss)
        for (auto loss : v)
            sortedLosses.push_back(loss);
    std::sort(sortedLosses.begin(), sortedLosses.end()); // Ascending
}


double BMT::loss(vector<double> const & wEval,
                 vector<double> const & wOpt,
                 vector<int> const & states, DiscreteCMP const & cmp)
{
    double sup = -DBL_MAX;
    // TODO: Do this for all state action pairs in a demonstration instead?
    // Probably not though
    for (int s : states)
    {
        vector<double> phi = cmp.features(s);
        double vOpt = inner_product(phi.begin(), phi.end(), wEval.begin(), 0.0);
        double vEval = inner_product(phi.begin(), phi.end(), wOpt.begin(), 0.0);
        double diff = fabs(vOpt - vEval);
        if (diff > sup)
            sup = diff;
    }

    return sup;
}

double BMT::getLoss(int policy, int rewardFunction)
{
    return policyRewardLoss[policy][rewardFunction];
}

double beta(double ep_lower, double ep_upper)
{
    const static double c = 0.1;
    return (exp(-c*ep_lower) - exp(-c*ep_upper));
}

double BMT::getRewardProbability(int rewardFunction) // Psi(B | ep, pi)
{
    const int j = rewardFunction;
    double probabilitySum = 0.0;

    for (int k = 0; k < K; ++k)
        for (int i = 0; i < sortedLosses.size() - 1; ++i)
        {
            double ep_lower = sortedLosses[i];
            double ep_upper = sortedLosses[i+1];
            if (getLoss(k, j) < ep_lower)
            {
                int nEpOptimalRewardFunctions = 0;
                for (int l = 0; l < N; ++l)
                    if (getLoss(k,l) <= ep_lower)
                        ++nEpOptimalRewardFunctions;
                probabilitySum += beta(ep_lower, ep_upper)
                                    / (double) nEpOptimalRewardFunctions;
            }
        }

    return probabilitySum;
}

// Ax = b, returns x for rectangular systems
vector<double> solve_rect(vector<double> A, vector<double> b)
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
