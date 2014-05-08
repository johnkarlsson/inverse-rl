#include "DirichletPolicyPosterior.h"
#include "Policy.h"
#include <memory>
#include <gsl/gsl_randist.h>
// #include <gsl/gsl_rng.h>

using std::vector;
using std::unique_ptr;
using std::pair;

vector<unique_ptr<Policy>> DirichletPolicyPosterior::sampledPolicies
    = vector<unique_ptr<Policy>>();

gsl_rng * StateActionDirichlet::r_global = nullptr;

void StateActionDirichlet::sample(int s, double * output)
{
    vector<double> alphas(actions, 0);
    for (int a = 0; a < actions; ++a)
        alphas[a] = getAlpha(s, a);
    gsl_ran_dirichlet(r_global, actions, &alphas[0], output);
}

vector<double>& DirichletPolicyPosterior::SampledPolicy::getMultinomial(int s)
{
    // Lazy sampling from parent since |S| may be large
    auto it = stateMultinomials.find(s);
    if (it == stateMultinomials.end())
    {
        it = stateMultinomials.insert(
                {s, vector<double>(parent->actions, 0)}).first;
        parent->sample(s, &(it->second[0]));
    }
    return it->second;
}

vector<pair<int,double>>
    DirichletPolicyPosterior::SampledPolicy::probabilities(int s)
{
    vector<double>& multinomial = getMultinomial(s);

    vector<pair<int,double>> output(parent->actions);
    for (int a = 0; a < parent->actions; ++a)
        output[a] = {a, multinomial[a]};

    return output;
}

Policy& DirichletPolicyPosterior::samplePolicy()
{
    Policy * pi = new DirichletPolicyPosterior::SampledPolicy(this);
    sampledPolicies.push_back(unique_ptr<Policy>(pi));
    return *sampledPolicies.back();
}

double DirichletPolicyPosterior::getAlpha(int s, int a)
{
    return prior.getAlpha(s, a) + getActionCounts(s)[a];
}

vector<int>& DirichletPolicyPosterior::getActionCounts(int s)
{
    auto it = stateActionCounts.find(s);
    if (it == stateActionCounts.end())
        it = stateActionCounts.insert({s, vector<int>(actions, 0)}).first;
    return it->second;
}
