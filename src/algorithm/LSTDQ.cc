#include "LSTDQ.h"
#include "../model/Policy.h"
#include "../model/DiscreteMDP.h"

#include <ctime>

Transition::Transition(int _s, int _a, int _s2, double _r)
    : s(_s), a(_a), s2(_s2), r(_r)
{}
// Transition::Transition(int _s, int _a)
//     : s(_s), a(_a)
// {}

vector<double> LSTDQ::solve(int nFeatures, int nSamples,
                            vector<double>& phi, vector<double>& td,
                            vector<double>& b)
{
    bool print = false;

    int k = nFeatures;
    int n = nSamples;
    assert(phi.size() == k*n);
    assert(td.size() == n*k);
    assert(b.size() == k);

    gsl_matrix_view _phi = gsl_matrix_view_array(phi.data(), k, n);
    gsl_matrix_view _td = gsl_matrix_view_array(td.data(), n, k);
    // gsl_matrix_view _A = gsl_matrix_view_array(A.data(), k, k);
    gsl_vector_view _b = gsl_vector_view_array(b.data(), k);
    gsl_matrix *_A = gsl_matrix_alloc(k, k);

    /* A = phi td */
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
                   1.0, &_phi.matrix, &_td.matrix,
                   // 0.0, &_A.matrix);
                   0.0, _A);

    for (int i = 0; i < k; ++i)
        _A->data[i*k + i] += 1e-6;

    if (print)
    {
        for (int i = 0; i < k; ++i)
        {
            for (int j = 0; j < k; ++j)
                cout << _A->data[i*k + j] << " ";
            cout << endl;
        }
        for (int i = 0; i < k; ++i)
        {
            cout << _b.vector.data[i] << endl;
        }
    }

    /* Compute w = A^(-1) b */
    gsl_permutation * p = gsl_permutation_alloc(k);
    gsl_vector *w = gsl_vector_alloc(k);
    int s;
    // gsl_linalg_LU_decomp(&_A.matrix, p, &s);
    gsl_linalg_LU_decomp(_A, p, &s);
    // gsl_linalg_LU_solve(&_A.matrix, p, &_b.vector, w);
    gsl_linalg_LU_solve(_A, p, &_b.vector, w);

    vector<double> output(w->data, w->data + k);

    assert(output.size() == k);

    gsl_permutation_free(p);
    gsl_matrix_free(_A);
    gsl_vector_free(w);


    return output;
}

vector<double> LSTDQ::lstdq(vector<Demonstration> const & D, Policy& pi,
                            DiscreteMDP const & mdp, bool withModel)
{
    int n = 0;
    for (auto d : D)
        n += d.size();
    const int k = mdp.cmp->nFeatures();
    const double gamma = mdp.gamma;

    vector<double> phi(k*n);
    vector<double> td(n*k);
    vector<double> b(k);

    int i = 0;
    for (vector<Transition> demonstration : D)
    {
        for (int t = 0; t < demonstration.size(); ++t, ++i)
        {
            Transition T = demonstration[t];

            // The "with a model" version of LSTDQ below assumes there are
            // successor states to iterate over, so we ignore terminal states.
            if (mdp.cmp->isTerminal(T.s)) // However not occurring in practice
                continue;

            vector<double> features = mdp.cmp->features(T.s, T.a);

            vector<pair<int,double>> transitions;
            if (withModel)
                // Transitions are given by the model and averaged over.
                transitions
                    = mdp.cmp->kernel->getTransitionProbabilities(T.s, T.a);
            else
                // Just one sample from the data
                transitions = {{T.s2, 1}};

            /*
             *  Average featuresPi = phi(s2, pi(s2)) and R(s2) over transition
             *  probabilities ("with a model"-version).
             */
            vector<double> featuresPi(k, 0); // Avg.
            double reward = 0; // Avg.
            for (auto tr : transitions)
            {
                int s2 = tr.first;
                double transitionProbability = tr.second;

                // Average phi(s2, pi(s2)) over policy probabilities (regardless
                // of LSTDQ version).
                if (!mdp.cmp->isTerminal(s2)) // Otherwise featuresPi == 0
                {
                    auto actionProbabilities = pi.probabilities(s2);
                    for (pair<int,double> ap : actionProbabilities)
                    {
                        double actionProbability = ap.second;
                        // Note that features(s,a) is in itself an average but over
                        // transition probabilities.
                        auto phi_a = mdp.cmp->features(s2, ap.first);
                        for (int j = 0; j < k; ++j)
                            featuresPi[j] +=   transitionProbability
                                             * actionProbability
                                             * phi_a[j];
                    }
                }

                reward += transitionProbability * mdp.getReward(s2);
            }

            for (int j = 0; j < k; ++j)
            {
                // phi[i * k + j] = features[j];
                phi[j * n + i] = features[j];// / (double) n;
                // if (terminal)
                //     td[i * k + j] = features[j];
                // else
                    td[i * k + j] = (features[j] - gamma * featuresPi[j]);
                if (reward != 0)
                    b[j] += reward * features[j];// / (double) n;
            }
        }
    }

    //cout << "~lstdq()" << endl;
    return solve(k, n, phi, td, b);
}


DeterministicPolicy LSTDQ::lspi(vector<Demonstration> const & D,
                                DiscreteMDP const & mdp, bool print,
                                double epsilon)
{
    return LSTDQ::lspi(D, mdp, vector<double>(mdp.cmp->nFeatures(),0), print,
                       epsilon);
}

DeterministicPolicy LSTDQ::lspi(vector<Demonstration> const & D,
                                DiscreteMDP const & mdp,
                                vector<double> const & initialWeights,
                                bool print, double epsilon)
{
    assert(initialWeights.size() == mdp.cmp->nFeatures());

    clock_t t = clock();

    DeterministicPolicy pi(mdp.cmp, initialWeights);

    const int MAX_ITERATIONS = 100;
    int i;
    for (i = 0; i < MAX_ITERATIONS; ++i)
    {
        vector<double> weights = lstdq(D, pi, mdp);
        bool done = true;
        for (int j = 0; j < weights.size(); ++j)
            if (fabs(weights[j] - pi.getWeights()[j]) > epsilon)
                done = false;
        pi.setWeights(weights);
        if (done)
            break;
    }
    if (i == MAX_ITERATIONS) // print regardless of print bool
        cout << "LSPI ended after MAX_ITERATIONS = " << MAX_ITERATIONS;
    else if (print)
        cout << "LSPI converged after " << i << " iterations";

    float seconds = ((float)(clock() - t))/CLOCKS_PER_SEC;
    if (print)
        cout << " in " << seconds << " seconds." << endl;

    return pi;
}

void test_lstdq()
{
    srand((unsigned)time(NULL));

    int K = 1; // O(K)
    const int n = 10000; // O(n)
    const int k = 20;
    bool print = true;

    for (int iteration = 0; iteration < K; ++iteration)
    {
        vector<double> phi(k*n);
        vector<double> td(n*k);
        // vector<double> A(k*k);
        vector<double> b(k);
        for (int i = 0; i < k; ++i)
            b[i] = r();
        for (int i = 0; i < k*n; ++i)
        {
            phi[i] = r();
            td[i] = r();
        }

        vector<double> w = LSTDQ::solve(k, n, phi, td, b);
        if (print)
        {
            cout << "LSTDQ w* = ";
            for (int i = 0; i < k; ++i)
                cout << w[i] << " ";
            cout << endl;
        }
    }
}
