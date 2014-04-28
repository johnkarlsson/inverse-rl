#include "RandomTransitionKernel.h"
#include <gsl/gsl_randist.h>
#include <iostream>
#include <iomanip>
#include <vector>

using std::vector;

void RandomTransitionKernel::test_dirichlet()
{
    double sum = 0;
    vector<double> theta = sample_multinomial();
    std::cout << "Sampled multinomial: ";
    for (int i = 0; i < states; ++i)
    {
        std::cout << theta[i] << " ";
        sum += theta[i];
    }
    std::cout << std::endl;
    std::cout << "Sum = " << sum << std::endl;
}

vector<double> RandomTransitionKernel::sample_multinomial()
{
    vector<double> multinomial(states, 0);
    gsl_ran_dirichlet(r_global, states, &alphas[0], &multinomial[0]);

    return multinomial;
}

RandomTransitionKernel::RandomTransitionKernel(int _states, int _actions)
    : TabularTransitionKernel(_states, _actions),
      alphas(_states, alpha)
{
    r_global = gsl_rng_alloc(gsl_rng_default);
    gsl_rng_set(r_global, (unsigned)time(NULL));

    for (int s = 0; s < states; ++s)
        for (int a = 0; a < actions; ++a)
        {
            vector<double> probabilities = sample_multinomial();
            for (int s2 = 0; s2 < states; ++s2)
                setTransitionProbability(s, a, s2, probabilities[s2]);
        }
}
