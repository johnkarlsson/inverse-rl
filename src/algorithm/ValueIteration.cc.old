#include "ValueIteration.h"
#include <vector>
#include <cfloat>

ValueIteration::ValueIteration(const DiscreteMDP * const _mdp)
    : mdp(_mdp)
{
    Q = std::vector<std::vector<double>>(
            mdp->cmp->states, std::vector<double>(mdp->cmp->actions, 0));
}

void ValueIteration::init(double epsilon)
{
    for (int s = 0; s < mdp->cmp->states; ++s)
    {
        for (int a = 0; a < mdp->cmp->actions; ++s)
        {
            double sum = 0;
            for (int s2 = 0; s2 < mdp->cmp->states; ++s2)
            {
                double p = mdp->cmp->getTransitionProbability(s, a, s2);
                double r = mdp->getReward(s2); // R(s,a,s')
                double max_a__Q_s2 = -DBL_MAX;
                for (int a2 = 0; a2 < mdp->cmp->actions; ++a2)
                    if (Q[s2][a2] > max_a__Q_s2)
                        max_a__Q_s2 = Q[s2][a2];
                sum += p * (r + mdp->gamma * max_a__Q_s2);
            }

            Q[s][a] = sum;
        }
    }
}
