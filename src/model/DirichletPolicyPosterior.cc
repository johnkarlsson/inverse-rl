#include "DirichletPolicyPosterior.h"
#include "Policy.h"
#include <memory>

std::vector<std::unique_ptr<Policy>> DirichletPolicyPosterior::sampledPolicies =std::vector<std::unique_ptr<Policy>>(); 
gsl_rng * DirichletPolicyPosterior::r_global = nullptr;

std::vector<std::pair<int,double>>
    DirichletPolicyPosterior::SampledPolicy::probabilities(int s) const
{
    return {{0,1}};
}

Policy& DirichletPolicyPosterior::samplePolicy()
{
    Policy * pi = new DirichletPolicyPosterior::SampledPolicy();
    sampledPolicies.push_back(std::unique_ptr<Policy>(pi));
    return *sampledPolicies.back();
}

std::vector<int>& DirichletPolicyPosterior::getStateActionCounts(int s)
{
    auto it = stateActionCounts.find(s);
    if (it == stateActionCounts.end())
        it = stateActionCounts.insert({s, std::vector<int>(actions, 0)}).first;
    return it->second;
}
