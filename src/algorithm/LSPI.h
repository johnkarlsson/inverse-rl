#ifndef LSPI_H
#define LSPI_H

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <random>
#include <iostream>
#include <assert.h>
#include <vector>

#include "../util.h"
#include "../model/Policy.h"
#include "../model/DiscreteMDP.h"

using namespace std;

class LSPI
{
    public:
        static vector<double> solve(int nFeatures, int nSamples,
                                    vector<double>& phi, vector<double>& td,
                                    vector<double>& b);
        static vector<double> lstdq(vector<Demonstration> const & D, Policy& pi,
                                    DiscreteMDP const & mdp,
                                    bool withModel = true);
        static DeterministicPolicy lspi(vector<Demonstration> const & D,
                                        DiscreteMDP const & mdp,
                                        vector<double> const & initialWeights,
                                        bool print = true,
                                        double epsilon = 1e-7,
                                        bool withModel = true);
        static DeterministicPolicy lspi(vector<Demonstration> const & D,
                                        DiscreteMDP const & mdp,
                                        bool print = true,
                                        double epsilon = 1e-7,
                                        bool withModel = true);
};

void test_lstdq();

#endif
