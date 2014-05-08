#include "LSTDQ.h"
#include "../model/Policy.h"
#include "../model/DiscreteMDP.h"

Transition::Transition(int _s, int _a, int _s2, double _r)
    : s(_s), a(_a), s2(_s2), r(_r)
{}

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
                            DiscreteMDP const & mdp)
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
            // TODO: Makes sense?
            bool terminal = (t == (demonstration.size() - 1));

            vector<double> features = mdp.cmp->features(T.s, T.a);
            // vector<double> features = mdp.cmp->features(T.s2);

            // Average phi(s2, pi(s2)) over policy probabilities.
            // Note that features(s,a) is also an average but over transition
            //  probabilities.
            vector<double> featuresPi(k, 0);
            auto actionProbabilities = pi.probabilities(T.s2);
            for (pair<int,double> ap : actionProbabilities)
            {
                auto phi_a = mdp.cmp->features(T.s2, ap.first);
                for (int j = 0; j < k; ++j)
                    featuresPi[j] += ap.second * phi_a[j];
            }

            for (int j = 0; j < k; ++j)
            {
                // phi[i * k + j] = features[j];
                phi[j * n + i] = features[j];// / (double) n;
                if (terminal)
                    td[i * k + j] = features[j];
                else
                    td[i * k + j] = (features[j] - gamma * featuresPi[j]);
                if (T.r != 0)
                    b[j] += T.r * features[j];// / (double) n;
            }
        }
    }

    return solve(k, n, phi, td, b);
}


DeterministicPolicy LSTDQ::lspi(vector<Demonstration> const & D,
                                DiscreteMDP const & mdp,
                                double epsilon)
{
    return LSTDQ::lspi(D, mdp, vector<double>(mdp.cmp->nFeatures(),0), epsilon);
}

DeterministicPolicy LSTDQ::lspi(vector<Demonstration> const & D,
                                DiscreteMDP const & mdp,
                                vector<double> const & initialWeights,
                                double epsilon)
{
    assert(initialWeights.size() == mdp.cmp->nFeatures());

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
    if (i == MAX_ITERATIONS)
        cout << "LSPI ended after MAX_ITERATIONS = " << MAX_ITERATIONS << endl;
    else
        cout << "LSPI converged after " << i << " iterations" << endl;

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