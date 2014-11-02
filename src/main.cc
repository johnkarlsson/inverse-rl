#include <iostream>
#include <cmath>
#include <iterator>
#include <algorithm>
#include <vector>
#include <set>
#include <sstream>
#include <cfloat>
#include <limits>
#include <chrono>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "model/DiscreteCMP.h"
#include "model/random_mdp/RandomTransitionKernel.h"
#include "model/TransitionKernel.h"
#include "model/TicTacToe/TicTacToeTransitionKernel.h"
#include "model/TicTacToe/TicTacToeMDP.h"
#include "model/TicTacToe/TicTacToeCMP.h"
#include "model/TicTacToe/OptimalTTTPolicy.h"
#include "model/TicTacToe/RandomTTTPolicy.h"
#include "model/random_mdp/RandomCMP.h"
#include "model/random_mdp/RandomMDP.h"
#include "model/DirichletPolicyPosterior.h"
#include "model/Policy.h"
#include "algorithm/LSPI.h"
#include "algorithm/BMT.h"

#include "util.h"

using namespace std;

void run_ttt();
void run_random_mdp();
extern void test_valueiteration();

static const std::string default_console = "\033[0m";

int main(int argc, const char *argv[])
{
    run_ttt();
    // run_random_mdp();
    // test_valueiteration();
}

void printExpertScores(vector<DeterministicPolicy> experts, int N_EXPERTS,
                       DiscreteMDP& mdp, int nWinRateDemonstrations)
{
    cout << "Scores for the experts true policies:" << endl;
    for (int i = 0; i < N_EXPERTS; ++i)
    {
        double expertScore
            = getAverageOptimalUtility(experts[i], mdp, 0,
                                       nWinRateDemonstrations, -1);
            cout << "\t{ " << expertScore << " }\t (expert " << i << ")"
                 << endl;
    }
}

// Tic-tac-toe
void run_ttt()
{
    srand((unsigned)time(NULL));

    /************************************************
    * * * * * * * * * Main parameters * * * * * * * *
    ************************************************/
    const int    N_EXPERT_PLAYOUTS        =       23;
    // const int    N_EXPERT_PLAYOUTS        =        1;
    const int    N_N_EXPERT_PLAYOUTS[N_EXPERT_PLAYOUTS]
    = {20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180,
       200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000};
    const int    N_REWARD_FUNCTIONS       =       60;
    const int    N_POLICY_SAMPLES         =       20;
    const int    N_EXPERTS                =        3;
    const double EXPERT_OPTIMALITY_PRIORS[N_EXPERTS]
                           // = {8.0, 8.0, 8.0, 8.0};
                                = {20.0, 20.0, 20.0};
    /************************************************
     ***********************************************/
    /************************************************
    * * * * * * Semi static parameterrs * * * * * * *
    ************************************************/
    const bool PRINT_DEBUG               =      true;
    const bool useSum                    =      true;
    const bool useModel                  =     false;
    const int N_Q_APPROXIMATION_PLAYOUTS =         1;
    const int T_Q_APPROXIMATION_HORIZON  =        -1;
    const int nWinRateDemonstrations     =     50000;
    /************************************************
     ***********************************************/

    // MDP setup
    // const double gamma = 1;
    TicTacToeTransitionKernel kernel = TicTacToeTransitionKernel(3);
    TicTacToeCMP cmp(&kernel);
    TicTacToeMDP mdp(&cmp, true); // True rewards
    TicTacToeMDP mutableMdp(&cmp, false); // Feature rewards
    RandomTTTPolicy randomPolicy(&cmp, true); // Optimal terminal play
    RandomTTTPolicy randomRandomPolicy(&cmp, false); // Random terminal play
    OptimalTTTPolicy optimalPolicy(&cmp);

    // Demonstrations for LSTDQ only
    cout << "Generating LSTDQ demonstrations..." << endl;
    vector<Demonstration> lstdqDemonstrations
        = generateDemonstrations(mdp, { &randomPolicy } , -1,
                5000, // Number of playouts
                      // Should be high so that V_rho^* is accurate
                0, false);
    set<int> uniqueStates;
    for (Demonstration demo : lstdqDemonstrations)
        for (Transition tr : demo)
            uniqueStates.insert(tr.s);
    cout << uniqueStates.size() << " unique states in "
         << lstdqDemonstrations.size() << " demonstrations..." << endl;

    cout << "Creating experts from sampled reward functions" << endl;
    vector<Demonstration> rewardDemonstrations
        = generateDemonstrations(mdp, { &randomPolicy } , -1,
                                 200, // Number of playouts
                                 0, // Initial state
                                 false); // Print
    auto rf = sampleRewardFunctions(1,//;N_REWARD_FUNCTIONS,
                                    N_Q_APPROXIMATION_PLAYOUTS,
                                    T_Q_APPROXIMATION_HORIZON,
                                    rewardDemonstrations,
                                    randomPolicy,
                                    // optimalPolicy,
                                    mdp)[0];

    vector<vector<double>> trueRewardFunctions;
    vector<DeterministicPolicy> experts;
    for (int m = 0; m < N_EXPERTS; ++m)
    {
        vector<double> expertRf(rf);
        switch (m)
        {
            case 0:
                for (int f = 0; f < cmp.nFeatures(); ++f)
                {
                    if (
                            f != TicTacToeCMP::FEATURE_DOUBLETS_X
                         && f != TicTacToeCMP::FEATURE_DOUBLETS_O
                         && f != TicTacToeCMP::FEATURE_CROSSPOINTS_X
                         && f != TicTacToeCMP::FEATURE_CROSSPOINTS_O
                         // && f != TicTacToeCMP::FEATURE_SINGLETS_X
                         && f != TicTacToeCMP::FEATURE_SINGLETS_O
                         && f != TicTacToeCMP::FEATURE_CORNERS_X
                         && f != TicTacToeCMP::FEATURE_CORNERS_O
                         // && f != TicTacToeCMP::FEATURE_CORNERS_O
                         && f != TicTacToeCMP::FEATURE_CENTER
                       )
                        expertRf[f] = 0;
                }
                break;

            case 1:
                for (int f = 0; f < cmp.nFeatures(); ++f)
                {
                    if (
                            f != TicTacToeCMP::FEATURE_FORKS_X
                         // && f != TicTacToeCMP::FEATURE_FORKS_O
                         && f != TicTacToeCMP::FEATURE_CENTER
                       )
                        expertRf[f] = 0;
                }
                break;

            case 2:
                for (int f = 0; f < cmp.nFeatures(); ++f)
                {
                    if (
                            f != TicTacToeCMP::FEATURE_SINGLETS_X
                         // && f != TicTacToeCMP::FEATURE_SINGLETS_O
                         && f != TicTacToeCMP::FEATURE_CORNERS_O
                         && f != TicTacToeCMP::FEATURE_CORNERS_X
                         && f != TicTacToeCMP::FEATURE_FORKS_O
                         && f != TicTacToeCMP::FEATURE_CENTER
                       )
                        expertRf[f] = 0;
                }
                break;
            default:
                break;
        }
        trueRewardFunctions.push_back(expertRf);
        mutableMdp.setRewardWeights(expertRf);
        auto expertPolicy = LSPI::lspi(lstdqDemonstrations, mutableMdp);
        experts.push_back(expertPolicy);
    }

    printExpertScores(experts, N_EXPERTS, mdp, nWinRateDemonstrations);


    // Evaluate programmatic optimal policy (LSTDQ)
    auto optWeights = LSPI::lstdq(lstdqDemonstrations, optimalPolicy, mdp);
    cout.precision(numeric_limits<double>::digits10 - 12);
    cout << "Optimal weights X(s,d,t,x,c,f) O(s,d,t,x,c,f) : " << endl;
    for (double d : optWeights)
        cout << fixed << d << "\t";
    cout << endl;

    // Evaluate random policy (LSTDQ)
    auto ranWeights = LSPI::lstdq(lstdqDemonstrations, randomPolicy, mdp);
    cout.precision(numeric_limits<double>::digits10 - 12);
    cout << "Random weights X(s,d,t,x,c,f) O(s,d,t,x,c,f) : " << endl;
    for (double d : ranWeights)
        cout << fixed << d << "\t";
    cout << endl;

    // Global set of reward functions
    cout << "Sampling " << N_REWARD_FUNCTIONS << " reward functions..." << endl;
    // Find rewards with some noise, for higher variance... (SoftmaxPolicy)
    // SoftmaxPolicy rewardPolicy(&cmp, lspiPolicy.getWeights(), 20.0);
    vector<vector<double>> rewardFunctions;
    for (int i = 0; i < N_REWARD_FUNCTIONS; ++i)
    {
        /*
         * Attempt to sample with as much noise as possible (low # playouts)
         */
        vector<Demonstration> rewardDemonstrations
            = generateDemonstrations(mdp, { &randomPolicy } , -1,
                    10, // Number of playouts
                    0, // Initial state
                    false); // Print
        auto rf = sampleRewardFunctions(N_REWARD_FUNCTIONS,
                                        N_Q_APPROXIMATION_PLAYOUTS,
                                        T_Q_APPROXIMATION_HORIZON,
                                        rewardDemonstrations,
                                        randomPolicy,
                                        // optimalPolicy,
                                        mdp)[0];
                                 // lspiPolicy, mdp);
                                 // rewardPolicy, mdp);
        /*
         * Resampling for when low # playouts gives unstable/diverging results.
         * (all of this is only due to the fact that TTT is a too easy problem)
         */
        static auto invalid_reward = [](float f)
        { return isnan(f) || isinf(f) || f < -10 || f > 10; };
        bool foundInvalidReward =
            (find_if(rf.begin(), rf.end(), invalid_reward) != rf.end());
        if (foundInvalidReward)
            --i;
        else
            rewardFunctions.push_back(rf);
    }

    if (PRINT_DEBUG)
    {
        cout.precision(numeric_limits< double >::digits10 - 12);
        cout << "\tSampled reward functions:" << endl;
        int i = 0;
        for (auto rf : rewardFunctions)
        {
            cout << "\t (" << i++ << ")\t";
            for (auto r : rf)
                cout << fixed << r << "\t";
            cout << endl;
        }
    }

    DeterministicPolicy lspiPolicy = LSPI::lspi(lstdqDemonstrations, mdp);
    // TODO: Save/read on file?
    // Global set of corresponding optimal policies
    cout << "Creating " << N_REWARD_FUNCTIONS
         << " corresponding optimal policies..." << endl;
    vector<DeterministicPolicy> optimalPolicies;
    for (int i = 0; i < N_REWARD_FUNCTIONS; ++i)
    {
        mutableMdp.setRewardWeights(rewardFunctions[i]);
        optimalPolicies.push_back(LSPI::lspi(lstdqDemonstrations, mutableMdp,
                                              false));
    }

    if (PRINT_DEBUG)
    {
        cout.precision(numeric_limits< double >::digits10 - 12);
        cout << "\tTrue { score } and optimal weights:" << endl;
        // cout << "\t{ " << programmaticOptimalScore << " }\t";
        cout << "\t{ " << "<disabled>" << " }\t";
        for (auto w : lspiPolicy.getWeights())
            cout << fixed << w << "\t";
        cout << endl;
        // cout << "\t{ " << randomScore << " }\t (random)" << endl;
        // cout << "\t{ " << randomRandomScore << " }\t (random random)" << endl;
        cout << "\t{ score } and optimal weights of the optimal policies"
             << " (from sampled reward functions):" << endl;
        for (int i = 0; i < N_REWARD_FUNCTIONS; ++i)
        {
            cout << " (" << i << ")\t" << fixed
                 // << "{ " << optimalWinRates[i] << " }\t";
                 << "{ " << "<disabled>" << " }\t";
            for (auto w : optimalPolicies[i].getWeights())
                cout << fixed << w << "\t";
            cout << endl;
        }
    }


    // Variance is high between runs, so to demonstrate convergence, create one
    // large dataset per expert and later use subsets of increasing size.
    const int MAX_EXPERT_PLAYOUTS = N_N_EXPERT_PLAYOUTS[N_EXPERT_PLAYOUTS-1];
    vector<vector<Demonstration>> expertBaseDemonstrations;
    for (int i = 0; i < N_EXPERTS; ++i)
    {
        auto demonstrations
            = generateDemonstrations(mdp, {&experts[i]},
                                     -1, /* horizon = playout */
                                     MAX_EXPERT_PLAYOUTS,
                                     0, /* initialState */
                                     false); /* No print */
        expertBaseDemonstrations.push_back(demonstrations);
    }

    for (int iteration = 0; iteration < N_EXPERT_PLAYOUTS; ++iteration)
    {
        const int N_EXPERT_DEMONSTRATIONS = N_N_EXPERT_PLAYOUTS[iteration];

        // Generate demonstrations
        cout << "Generating " << N_EXPERTS << " expert demonstrations of size "
             << N_EXPERT_DEMONSTRATIONS << "..." << endl;
        vector<vector<Demonstration>> expertDemonstrations;
        for (int i = 0; i < N_EXPERTS; ++i)
        {
            auto it = expertBaseDemonstrations[i].begin();
            vector<Demonstration> demonstrations(it,
                                                 it + N_EXPERT_DEMONSTRATIONS);
            assert(demonstrations.size() == N_EXPERT_DEMONSTRATIONS);
            // auto demonstrations
            //     = generateDemonstrations(mdp, {&experts[i]},
            //                              -1, /* horizon = playout */
            //                              N_EXPERT_DEMONSTRATIONS,
            //                              0, /* initialState */
            //                              false); /* No print */
            expertDemonstrations.push_back(demonstrations);
        }

        // Initialize policy posteriors given expert data
        cout << "Initializing " << N_EXPERTS << " policy posteriors..." << endl;
        vector<DirichletPolicyPosterior> posteriorPolicies;
        for (int i = 0; i < N_EXPERTS; ++i)
        {
            double priorAlpha = 0.1;
            SoftmaxDirichletPrior policyPrior(9, &cmp, priorAlpha);
            posteriorPolicies.push_back(
                DirichletPolicyPosterior(policyPrior, expertDemonstrations[i]));
        }

        // Sample K policies from the posterior of each expert
        cout << "Sampling " << N_POLICY_SAMPLES << " policies from "
             << N_EXPERTS << " experts..." << endl;
        vector<vector<Policy*>> sampledPolicies(N_EXPERTS);
        for (int i = 0; i < N_EXPERTS; i++)
            for (int k = 0; k < N_POLICY_SAMPLES; ++k)
                sampledPolicies[i].push_back(
                    &posteriorPolicies[i].samplePolicy());

        // Initialize BMT calculations for each expert
        cout << "Initializing " << N_EXPERTS << " BMT objects..." << endl;
        vector<BMT> bmts;
        for (int i = 0; i < N_EXPERTS; ++i)
        {
            BMT bmt(mutableMdp,
                    // Using expert demonstrations instead of
                    // lstdqDemonstrations since this gives better support for
                    // the posterior policy. See thesis.
                    expertDemonstrations[i],
                    rewardFunctions,
                    optimalPolicies,
                    sampledPolicies[i],
                    EXPERT_OPTIMALITY_PRIORS[i],
                    useModel,
                    useSum);
            bmts.push_back(bmt);
        }

        /**
          * Calculation of final loss...
          */
        vector<double> probabilityProducts(N_REWARD_FUNCTIONS, 1);
        for (BMT& bmt : bmts)
        {
            vector<double> probabilities = bmt.getNormalizedRewardProbabilities();
            for (int i = 0; i < N_REWARD_FUNCTIONS; ++i)
                probabilityProducts[i] *= probabilities[i];
        }

        vector<double> combinedMeanRewardFunction(cmp.nFeatures(), 0);
        for (int i = 0; i < N_EXPERTS; ++i)
        {
            BMT& bmt = bmts[i];
            int K = N_POLICY_SAMPLES;
            int N = N_REWARD_FUNCTIONS;
            if (PRINT_DEBUG) // loss matrix
            {
                cout.precision(numeric_limits< double >::digits10 - 12);
                cout << "Loss matrix" << endl;
                for (int k = 0; k < K; ++k)
                {
                    for (int j = 0; j < N; ++j)
                    {
                        cout << fixed << bmt.getLoss(k,j) << "\t";
                    }
                    cout << endl;
                }
            }
            cout.precision(numeric_limits< double >::digits10 - 12);
            vector<double> rewardProbabilities
                = bmt.getNormalizedRewardProbabilities();
            cout << "Reward function probabilities:" << endl;
            for (int rr = 0; rr < N_REWARD_FUNCTIONS; ++rr)
                cout << rr << "\t";
            cout << endl;
            for (auto rewardProb : rewardProbabilities)
                cout << fixed << rewardProb << "\t";
            cout << endl;
            cout << endl;

            // Average reward function
            vector<double> meanRewardFunction(cmp.nFeatures(), 0);
            for (int j = 0; j < N_REWARD_FUNCTIONS; ++j)
            {
                double p = rewardProbabilities[j];
                for (int k = 0; k < meanRewardFunction.size(); ++k)
                {
                    meanRewardFunction[k] += p * rewardFunctions[j][k];
                    combinedMeanRewardFunction[k] += p / ((double) N_EXPERTS)
                                                     * rewardFunctions[j][k];
                }
            }
            cout << "Expected reward function:" << endl;
            for (auto r : meanRewardFunction)
                cout << r << "\t";
            cout << endl;

            // Take expected reward function m --> get optimal policy m
            // Compare policy m weights with true expert V_m
            mutableMdp.setRewardWeights(meanRewardFunction);
            auto meanRFOptimalPolicy = LSPI::lspi(expertDemonstrations[i],
                                                  mutableMdp);
            mutableMdp.setRewardWeights(trueRewardFunctions[i]); // update
            auto meanRFOptimalPolicyInTrueMDPWeights
                = LSPI::lstdq(expertDemonstrations[i], meanRFOptimalPolicy,
                              mutableMdp);
            auto MDP_m_optimalWeights =  // Update:
                experts[i].getWeights(); //   expert[i] is OptimalPolicy
            double meanRFOptimalPolicyLoss
                = bmts[i].loss(meanRFOptimalPolicyInTrueMDPWeights,
                               MDP_m_optimalWeights,
                               useSum);
            double meanRFOptimalPolicyScore
                = getAverageOptimalUtility(meanRFOptimalPolicy, mdp, 0,
                                           nWinRateDemonstrations, -1);
            // Compared to true MDP_m == rho_m == expert_m.weights
            cout << "\033[0;34mMEAN RF POLICY* LOSS IN MDP_m: "
                 << default_console
                 << meanRFOptimalPolicyLoss << endl;
            cout << "\033[0;34mMEAN RF POLICY* SCORE IN MDP: "
                 << default_console
                 << meanRFOptimalPolicyScore << endl;
            cout << endl;
        }

        // Combined mean
        // Calculate ~optimal policy for combined mean
        mutableMdp.setRewardWeights(combinedMeanRewardFunction);
        auto combinedMeanRFOptimalPolicy = LSPI::lspi(lstdqDemonstrations,
                                                      mutableMdp);

        double combinedMeanRFOptimalPolicyScore
            = getAverageOptimalUtility(combinedMeanRFOptimalPolicy, mdp, 0,
                                       nWinRateDemonstrations, -1);
        cout << "\033[0;32m~~~ COMBINED SCORE ~~~\t"
             << default_console
             << combinedMeanRFOptimalPolicyScore
             << endl;
    }
}

void run_random_mdp()
{
    clock_t t = clock();
    srand((unsigned)time(NULL));

    /************************************************
    * * * * * * * * * Main parameters * * * * * * * *
    ************************************************/
    const int    N_EXPERT_DEMO_HORIZONS   =       11;
    const int    T_EXPERT_DEMO_HORIZONS[] =
        {1000, 1, 2, 10, 20, 50, 100, 150, 200, 500, 1000}; // < 5000 necessary
    const int    N_EXPERT_DEMONSTRATIONS  =        1;
    const int    N_REWARD_FUNCTIONS       =       20;
                                    //    =       20;          // A
    const int    N_POLICY_SAMPLES         =       10;
                                    //    =       30;          // A
    const int    N_EXPERTS                =        5;
    const double EXPERT_TEMPERATURES[]    =
                   // {0.001, 0.005, 0.01,  0.015, 0.02};      // A
                      {0.001, 0.001, 0.001, 0.001, 0.001};
    const double EXPERT_OPTIMALITY_PRIORS[]    =
                           // {10 ,  9.5, 8.0, 7.0, 6.0};      // A
                              {60 ,  60, 60 , 60, 60};
    /************************************************
     ***********************************************/
    /************************************************
    * * * * * * Semi static parameterrs * * * * * * *
    ************************************************/
    const bool PRINT_DEBUG                 =    true;
    const bool useSum                      =    true;
    const bool useModel                    =    true;
    const bool multipleTrueRewardFunctions =    true;
    /************************************************
     ***********************************************/

    // MDP setup
    const double gamma = 0.90;
    // const double gamma = 0.50;
    const int states = 20;
    const int actions = 5;
    RandomTransitionKernel kernel(states, actions);
    RandomCMP cmp(&kernel);
    RandomMDP mdp(&cmp, gamma);
    const vector<double> trueRewardFunction = mdp.getRewardWeights();

    // Demonstrations for LSTDQ only
    cout << "Generating LSTDQ demonstrations..." << endl;
    /**
      * Hack useModel
      */
    vector<Demonstration> lstdqDemonstrations;
    Demonstration d;
    for (int i = 0; i < states; ++i)
        d.push_back(Transition(i, 0,0,0));
    lstdqDemonstrations.push_back(d);
    set<int> uniqueStates;
    for (Demonstration demo : lstdqDemonstrations)
        for (Transition tr : demo)
            uniqueStates.insert(tr.s);

    // Optimal policy
    cout << "Calculating optimal policy (LSPI)..." << endl;
    DeterministicPolicy lspiPolicy = LSPI::lspi(lstdqDemonstrations, mdp,
                                                 true, 1e-7, useModel);

    // Global set of reward functions
    // Find rewards with some noise, for more variance... (SoftmaxPolicy)
    cout << "Sampling " << N_REWARD_FUNCTIONS << " reward functions..." << endl;
    /*
    SoftmaxPolicy rewardPolicy(&cmp, lspiPolicy.getWeights(), 20.0);
    vector<vector<double>> rewardFunctions = 
        sampleRewardFunctions(N_REWARD_FUNCTIONS, N_Q_APPROXIMATION_PLAYOUTS,
                              T_Q_APPROXIMATION_HORIZON, lstdqDemonstrations,
                              rewardPolicy, mdp);
    */
    vector<vector<double>> rewardFunctions;
    for (int i = 0; i < N_REWARD_FUNCTIONS; ++i)
    {
        if (false)
        {
            vector<double> rf(cmp.nFeatures(), 0);
            rf[i % cmp.nFeatures()] = 1;
            rewardFunctions.push_back(rf);
        }
        else
        {
            auto rf = kernel.sample_multinomial();
            assert(rf.size() == mdp.cmp->nFeatures());
            rewardFunctions.push_back(rf);
        }
    }
    if (PRINT_DEBUG)
    {
        cout.precision(numeric_limits< double >::digits10 - 12);
        cout << "\tTrue reward function:" << endl;
        cout << "\t\t";
        for (auto r : trueRewardFunction)
            cout << fixed << r << "\t";
        cout << endl;
        cout << "\tSampled reward functions:" << endl;
        int i = 0;
        for (auto rf : rewardFunctions)
        {
            cout << "\t (" << i++ << ")\t";
            for (auto r : rf)
                cout << fixed << r << "\t";
            cout << endl;
        }
    }

    // Global set of corresponding optimal policies
    cout << "Creating " << N_REWARD_FUNCTIONS
         << " corresponding optimal policies..." << endl;
    vector<DeterministicPolicy> optimalPolicies;
    for (int i = 0; i < N_REWARD_FUNCTIONS; ++i)
    {
        mdp.setRewardWeights(rewardFunctions[i]);
        optimalPolicies.push_back(LSPI::lspi(lstdqDemonstrations, mdp, false,
                                              1e-7, useModel));
    }
    mdp.setRewardWeights(trueRewardFunction); // Reset to original state.
    if (PRINT_DEBUG)
    {
        cout.precision(numeric_limits< double >::digits10 - 12);
        cout << "\tTrue Optimal weights:" << endl;
        cout << "\t\t";
        for (auto w : lspiPolicy.getWeights())
            cout << fixed << w << "\t";
        cout << endl;
        cout << "\tSampled reward functions' optimal policies' optimal weights:"
             << endl;
        int i = 0;
        for (auto pi : optimalPolicies)
        {
            cout << "\t (" << i++ << ")\t";
            for (auto w : pi.getWeights())
                cout << fixed << w << "\t";
            cout << endl;
        }
    }

    // Initializing our experts
    cout << "Initializing " << N_EXPERTS << " experts...";
    vector<vector<double>> trueRewardFunctions;
    vector<SoftmaxPolicy> experts;
    for (int i = 0; i < N_EXPERTS; ++i)
    {
        if (multipleTrueRewardFunctions)
        {
            vector<double> expertRf(trueRewardFunction);
            // Perturb expertRf
            static gsl_rng *r_global = NULL;
            if (r_global == NULL)
            {
                r_global = gsl_rng_alloc(gsl_rng_default);
                gsl_rng_set(r_global, (unsigned)time(NULL));
            }
            for (int f = 0; f < cmp.nFeatures(); ++f)
            {
                double r = gsl_ran_gaussian_ziggurat(r_global, 4.0);
                // expertRf[f] = (1 + r) * expertRf[f];
                expertRf[f] = expertRf[f] + r;
            }
            trueRewardFunctions.push_back(expertRf);
            mdp.setRewardWeights(expertRf);
            auto expertDeterministic = LSPI::lspi(lstdqDemonstrations, mdp);
            auto expert = SoftmaxPolicy(&cmp, expertDeterministic.getWeights(),
                                        EXPERT_TEMPERATURES[i]);
            mdp.setRewardWeights(trueRewardFunction);
            experts.push_back(expert);
        }
        else
        {
            experts.push_back(SoftmaxPolicy(&cmp, lspiPolicy.getWeights(),
                                            EXPERT_TEMPERATURES[i]));
            trueRewardFunctions.push_back(trueRewardFunction);
        }
    }
    auto optimalWeights = lspiPolicy.getWeights();

    // True expert losses w.r.t. true MDP
    cout << "Calculating losses for each expert w.r.t. true MDP (rf),"
         << " to be compared with \"final loss\" later:" << endl;
    mdp.setRewardWeights(trueRewardFunction);
    for (int i = 0; i < N_EXPERTS; ++i)
    {
        auto expertWeights = LSPI::lstdq(lstdqDemonstrations, experts[i], mdp);
        double loss = BMT::loss(expertWeights, optimalWeights, uniqueStates,
                                cmp, useSum);
        cout << "\t" << loss << endl;
    }

    // Variance is high between runs, so to demonstrate convergence, create one
    // large dataset per expert and later use subsets of increasing size.
    const int MAX_EXPERT_HORIZON
        = T_EXPERT_DEMO_HORIZONS[N_EXPERT_DEMO_HORIZONS-1];
    vector<vector<Demonstration>> expertBaseDemonstrations;
    for (int i = 0; i < N_EXPERTS; ++i)
    {
        auto demonstrations
            = generateDemonstrations(mdp, {&experts[i]},
                                     MAX_EXPERT_HORIZON, /* horizon = playout */
                                     N_EXPERT_DEMONSTRATIONS, /* nDemonstrations */
                                     -1, /* initialState uniform random */
                                     false); /* No print */
        expertBaseDemonstrations.push_back(demonstrations);
    }

    for (int iteration = 0; iteration < N_EXPERT_DEMO_HORIZONS; ++iteration)
    {
        const int T_EXPERT_DEMO_HORIZON = T_EXPERT_DEMO_HORIZONS[iteration];

        // Generate demonstrations
        cout << "Generating " << N_EXPERTS << " expert demonstrations of length"
             << T_EXPERT_DEMO_HORIZON << endl;
        vector<vector<Demonstration>> expertDemonstrations;
        for (int i = 0; i < N_EXPERTS; ++i)
        {
            auto it = expertBaseDemonstrations[i][0].begin();
            Demonstration demonstration(it, it + T_EXPERT_DEMO_HORIZON);
            assert(demonstration.size() == T_EXPERT_DEMO_HORIZON);
            // auto demonstrations
            //     = generateDemonstrations(mdp, {&experts[i]},
            //                              T_EXPERT_DEMO_HORIZON,
            //                              N_EXPERT_DEMONSTRATIONS,
            //                              -1, /* initialState uniform random */
            //                              false); /* No print */
            expertDemonstrations.push_back({demonstration});
        }

        // Initialize policy posteriors given expert data
        cout << "Initializing " << N_EXPERTS << " policy posteriors..." << endl;
        vector<DirichletPolicyPosterior> posteriorPolicies;
        for (int i = 0; i < N_EXPERTS; ++i)
        {
            SoftmaxDirichletPrior policyPrior(actions);
            posteriorPolicies.push_back(
                DirichletPolicyPosterior(policyPrior, expertDemonstrations[i]));
        }

        // Sample K policies from the posterior of each expert
        cout << "Sampling " << N_POLICY_SAMPLES << " policies from " << N_EXPERTS
             << " experts..." << endl;
        vector<vector<Policy*>> sampledPolicies(N_EXPERTS);
        for (int i = 0; i < N_EXPERTS; i++)
            for (int k = 0; k < N_POLICY_SAMPLES; ++k)
                sampledPolicies[i].push_back(&posteriorPolicies[i].samplePolicy());

        if (PRINT_DEBUG) // Lower priority than debug.
        {
            cout.precision(numeric_limits< double >::digits10 - 12);
            cout << "\tSampled policies state action probabilities (first "
                 << "policy for each expert only:" << endl;
            int i = 0;
            for (vector<Policy*> expertPolicies : sampledPolicies)
            {
                cout << "\t\tExpert " << i++ << ":";
                Policy * pi = expertPolicies[0]; // First only
                for (int s = 0; s < states; ++s)
                {
                    cout << "\t\ts" << s << ":\t";
                    for (auto pr : pi->probabilities(s))
                        cout << fixed << pr.second << "\t";
                }
                cout << endl;
            }
        }

        // Initialize BMT calculations for each expert
        cout << "Initializing " << N_EXPERTS << " BMT objects..." << endl;
        vector<BMT> bmts;
        for (int i = 0; i < N_EXPERTS; ++i)
        {
            BMT bmt(mdp, lstdqDemonstrations, rewardFunctions, optimalPolicies,
                    sampledPolicies[i], EXPERT_OPTIMALITY_PRIORS[i],
                    useModel, useSum);
            bmts.push_back(bmt);
        }

        /**
          * Calculation of final loss...
          */
        vector<double> probabilityProducts(N_REWARD_FUNCTIONS, 1);
        for (BMT& bmt : bmts)
        {
            vector<double> probabilities = bmt.getNormalizedRewardProbabilities();
            for (int i = 0; i < N_REWARD_FUNCTIONS; ++i)
                probabilityProducts[i] *= probabilities[i];
        }

        vector<double> combinedMeanRewardFunction(cmp.nFeatures(), 0);
        // Printing of loss matrices
        for (int i = 0; i < N_EXPERTS; ++i)
        {
            BMT& bmt = bmts[i];
            int K = N_POLICY_SAMPLES;
            int N = N_REWARD_FUNCTIONS;
            if (PRINT_DEBUG) // loss matrix
            {
                cout.precision(numeric_limits< double >::digits10 - 12);
                cout << "Loss matrix" << endl;
                for (int k = 0; k < K; ++k)
                {
                    for (int j = 0; j < N; ++j)
                    {
                        cout << fixed << bmt.getLoss(k,j) << "\t";
                    }
                    cout << endl;
                }
            }
            cout.precision(numeric_limits< double >::digits10 - 12);
            vector<double> rewardProbabilities
                = bmt.getNormalizedRewardProbabilities();
            cout << "Reward function probabilities:" << endl;
            for (int rr = 0; rr < N_REWARD_FUNCTIONS; ++rr)
                cout << rr << "\t";
            cout << endl;
            for (auto rewardProb : rewardProbabilities)
                cout << fixed << rewardProb << "\t";
            cout << endl;
            cout << endl;

            // Average reward function
            vector<double> meanRewardFunction(cmp.nFeatures(), 0);
            for (int j = 0; j < N_REWARD_FUNCTIONS; ++j)
            {
                double p = rewardProbabilities[j];
                for (int k = 0; k < meanRewardFunction.size(); ++k)
                {
                    meanRewardFunction[k] += p * rewardFunctions[j][k];
                    combinedMeanRewardFunction[k] += p / ((double) N_EXPERTS)
                                                    * rewardFunctions[j][k];
                }
            }
            cout << "Expected reward function:" << endl;
            for (auto r : meanRewardFunction)
                cout << r << "\t";
            cout << endl;

            // Take expected reward function m --> get ~optimal policy m
            mdp.setRewardWeights(meanRewardFunction);
            auto meanRFOptimalPolicy = LSPI::lspi(lstdqDemonstrations, mdp,
                                                   true, 1e-7, useModel);
            // Evaluate ~optimal policy m in true environment
            mdp.setRewardWeights(trueRewardFunctions[i]);
            vector<double> MDP_m_optimalWeights;
            if (multipleTrueRewardFunctions)
                MDP_m_optimalWeights = experts[i].getWeights();
            else
                MDP_m_optimalWeights = optimalWeights;
            auto meanRFOptimalPolicyInTrueMDPWeights
                = LSPI::lstdq(lstdqDemonstrations, meanRFOptimalPolicy,
                               mdp);
            // Calculate loss w.r.t. true optimalWeights
            double meanRFOptimalPolicyLoss
                // = BMT::loss(experts[i].getWeights(),
                = bmts[i].loss(meanRFOptimalPolicyInTrueMDPWeights,
                               MDP_m_optimalWeights,
                               useSum);
            // Compared to true MDP_m == rho_m == expert_m.weights
            cout << "MEAN RF POLICY* LOSS IN MDP_m: "
                 << meanRFOptimalPolicyLoss << endl;
            cout << endl;
        }

        // Combined mean
        // Calculate ~optimal policy for combined mean
        mdp.setRewardWeights(combinedMeanRewardFunction);
        auto combinedMeanRFOptimalPolicy = LSPI::lspi(lstdqDemonstrations,
                                                      mdp, true, 1e-7,
                                                      useModel);
        // Evaluate combined ~optimal policy in true environment
        mdp.setRewardWeights(trueRewardFunction);
        auto combinedMeanRFOptimalPolicyInTrueMDPWeights
            = LSPI::lstdq(lstdqDemonstrations, combinedMeanRFOptimalPolicy,
                           mdp);
        double combinedMeanRFOptimalPolicyLoss
            = BMT::loss(combinedMeanRFOptimalPolicyInTrueMDPWeights,
                        optimalWeights,
                        uniqueStates, cmp,
                        useSum);
        cout << "~~~ COMBINED LOSS ~~~\t" << combinedMeanRFOptimalPolicyLoss
             << endl;
    }

    float seconds = ((float)(clock() - t))/CLOCKS_PER_SEC;
    cerr << "Finished in " << seconds << " seconds." << endl;
}
