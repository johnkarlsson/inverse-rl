#ifndef LSTDQ_H
#define LSTDQ_H

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <random>
#include <iostream>
#include <assert.h>
#include <vector>

#include "../model/Policy.h"
#include "../model/DiscreteMDP.h"

using namespace std;

class Transition
{
    public:
        Transition() {};
        Transition(int _s, int _a, int _s2, double _r);
        Transition(int _s, int _a);
        int s, a;
    private:
        double r, s2; // TODO: Remove, get this from the MDP
};

typedef vector<Transition> Demonstration;

class LSTDQ
{
    public:
        static vector<double> solve(int nFeatures, int nSamples,
                                    vector<double>& phi, vector<double>& td,
                                    vector<double>& b);
        static vector<double> lstdq(vector<Demonstration> const & D, Policy& pi,
                                    DiscreteMDP const & mdp);
        static DeterministicPolicy lspi(vector<Demonstration> const & D,
                                        DiscreteMDP const & mdp,
                                        vector<double> const &
                                            initialWeights,
                                        double epsilon = 1e-6);
        static DeterministicPolicy lspi(vector<Demonstration> const & D,
                                        DiscreteMDP const & mdp,
                                        double epsilon = 1e-6);
};

inline double r() { return ((double) rand() / (double) RAND_MAX); }

void test_lstdq();

#endif
