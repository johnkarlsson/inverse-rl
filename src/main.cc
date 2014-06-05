#include <iostream>
#include "model/DiscreteCMP.h"
// #include "model/DiscreteMDP.h"
// #include "model/GridCMP.h"
// #include "model/gridworld/GridTransitionKernel.h"
#include "model/random_mdp/RandomTransitionKernel.h"
#include "model/TicTacToe/TicTacToeTransitionKernel.h"
#include "model/TransitionKernel.h"
#include "model/TicTacToe/TicTacToeMDP.h"
#include "model/TicTacToe/TicTacToeCMP.h"
#include "model/TicTacToe/OptimalTTTPolicy.h"
#include "model/TicTacToe/RandomTTTPolicy.h"
#include "model/random_mdp/RandomCMP.h"
#include "model/random_mdp/RandomMDP.h"
#include "model/DirichletPolicyPosterior.h"
#include "model/Policy.h"
#include "datageneration/RandomTTTData.h"
#include "algorithm/LSTDQ.h"
#include "algorithm/BMT.h"
#include "algorithm/ValueIteration.h"
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

using namespace std;

void test_discretecmp();
void test_valueiteration();
void test_tictactoecmp();
void test_tictactoetransitionkernel();
void test_lstdq_optpolicy();
void play_optimalTTTpolicy();
void test_randomPolicy();
void test_tictactoecmp_print(TicTacToeCMP& tttCmp);
std::pair<double,double> optimal_vs_random();
void compare_vi_qi(int cmp_size=10, double epsilon=0.001);
void test_dirichletPolicyPosterior();
void test_BMT();
void test_BMT2();
void normalize(vector<double>& v);
int sample(vector<pair<int, double>> transitionProbabilities);
vector<Demonstration> generateRandomMDPDemonstrations(RandomMDP& mdp, int len);
//vector<Demonstration> generateDemonstrations(DiscreteMDP& mdp, Policy& pi,
vector<Demonstration> generateDemonstrations(DiscreteMDP& mdp,
                                             vector<Policy*> policies,
                                             int demonstrationLength,
                                             int nDemonstrations = 1,
                                             int initialState = -1,
                                             bool print = true);
vector<vector<double>> sampleRewardFunctions(int nFunctions, int nPlayouts,
                                             int playoutHorizon,
                                             vector<Demonstration>&
                                                 lstdqDemonstrations,
                                             Policy& optimalPolicy,
                                             DiscreteMDP& mdp);
void test_generateTTT();
void test_BMT3();
void test_BMT4();
double getAverageOptimalUtility(Policy& optimalPolicy, DiscreteMDP& mdp,
                                int state, int nPlayouts, int T);

vector<double> wGlobal;
static const std::string default_console = "\033[0m";

int main(int argc, const char *argv[])
{
    if (false)
    {
        test_BMT4();
        return 0;
    }

    if (true)
    {
        test_BMT3();
        return 0;
    }

    if (true)
    {
        test_valueiteration();
        return 0;
    }
    if (true)
    {
        test_generateTTT();
        return 0;
    }

    if (true)
    {
        test_BMT2();
        return 0;
    }
    else
    {
        test_BMT();
        return 0;
    }
    // test_dirichletPolicyPosterior();
    // return 0;
    test_valueiteration();
    return 0;
    // test_discretecmp();
    test_tictactoetransitionkernel();
    test_tictactoecmp();
    test_lstdq();
    test_lstdq_optpolicy();
    test_randomPolicy();
    play_optimalTTTpolicy();
    // compare_vi_qi(30, 10);
    return 0;
}

vector<double> test_BWT2_sampleRewardFunction(int features)
{
    vector<double> output(features, 0);
    for (int i = 0; i < output.size(); ++i)
        output[i] = r();
    normalize(output);
    return output;
}

/*



    const double EXPERT_TEMPERATURES[]    =
                   {0.001, 0.005, 0.01, 0.015, 0.02};
    const double EXPERT_OPTIMALITY_PRIORS[]    =
                           {10 ,  9.5, 8.0, 7.0, 6.0};
    const bool PRINT_DEBUG               =      true;
    const bool useSum                    =      true;
    const bool useModel                  =      true;
    const int lstdDemonstrationLength    =       500;
*/

// OptimalPolicy bad 
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
/*
   Tc tac toe
   */
void test_BMT4()
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
    // const double REWARD_ALPHA          =        1;
    const double EXPERT_TEMPERATURES[N_EXPERTS]
                             = {0.001, 0.001, 0.001}; // Let all be low.
    const double EXPERT_OPTIMALITY_PRIORS[N_EXPERTS]
                           // = {8.0, 8.0, 8.0, 8.0};
                                = {20.0, 20.0, 20.0};
    /************************************************
     ************************************************/
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
     ************************************************/

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
    // for (auto r : rf)
    //     cout << r << " ";
    // cout << endl;
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
        auto expertPolicy = LSTDQ::lspi(lstdqDemonstrations, mutableMdp);
        experts.push_back(expertPolicy);
        // cout << "Calculating its score" << endl;
        // double expertScore = getAverageOptimalUtility(expertPolicy, mdp, 0,
        //                                               nWinRateDemonstrations,
        //                                               -1);
        // cout << "\t[{ " << expertScore << " }]"//\t (expert " << i << ")"
        //      << endl;
    }

    printExpertScores(experts, N_EXPERTS, mdp, nWinRateDemonstrations);


    // Evaluate programmatic optimal policy (LSTDQ)
    auto optWeights = LSTDQ::lstdq(lstdqDemonstrations, optimalPolicy, mdp);
    cout.precision(numeric_limits<double>::digits10 - 12);
    cout << "Optimal weights X(s,d,t,x,c,f) O(s,d,t,x,c,f) : " << endl;
    for (double d : optWeights)
        cout << fixed << d << "\t";
    cout << endl;

    // Evaluate random policy (LSTDQ)
    auto ranWeights = LSTDQ::lstdq(lstdqDemonstrations, randomPolicy, mdp);
    cout.precision(numeric_limits<double>::digits10 - 12);
    cout << "Random weights X(s,d,t,x,c,f) O(s,d,t,x,c,f) : " << endl;
    for (double d : ranWeights)
        cout << fixed << d << "\t";
    cout << endl;

    // double loss = BMT::loss(optWeights, ranWeights, uniqueStates,
    //                         cmp, true); // sum?

    // TODO: Save/read on file?
    // Global set of reward functions
    // Find rewards with some noise, for more variance... (SoftmaxPolicy)
    cout << "Sampling " << N_REWARD_FUNCTIONS << " reward functions..." << endl;
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
         * Workaround for when low # playouts gives unstable/diverging results.
         */
        static auto invalid_reward = [](float f)
        { return isnan(f) || isinf(f) || f < -10 || f > 10; };
        bool foundInvalidReward =
            (find_if(rf.begin(), rf.end(), invalid_reward) != rf.end());
        if (foundInvalidReward)
            --i;
        else
        {
            /*
            static gsl_rng *r_global = NULL;
            static const vector<double> alphas(cmp.nFeatures(),
                                               REWARD_ALPHA);
            vector<double> multinomial(cmp.nFeatures(), 0);
            if (r_global == NULL)
            {
                r_global = gsl_rng_alloc(gsl_rng_default);
                gsl_rng_set(r_global, (unsigned)time(NULL));
            }
            gsl_ran_dirichlet(r_global, cmp.nFeatures(), &alphas[0],
                              &multinomial[0]);
            // for (int f = 0; f < cmp.nFeatures(); ++f)
            //     rf[f] *= multinomial[f];
            */
            rewardFunctions.push_back(rf);
        }
        // int retain = i % cmp.nFeatures(); // The feature to retain
        // for (int x = 0; x < cmp.nFeatures(); ++x)
        //     if (x != retain)
        //         rewardFunctions.back()[x] = 0;
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

    DeterministicPolicy lspiPolicy = LSTDQ::lspi(lstdqDemonstrations, mdp);
    // TODO: Save/read on file?
    // Global set of corresponding optimal policies
    cout << "Creating " << N_REWARD_FUNCTIONS
         << " corresponding optimal policies..." << endl;
    vector<DeterministicPolicy> optimalPolicies;
    for (int i = 0; i < N_REWARD_FUNCTIONS; ++i)
    {
        mutableMdp.setRewardWeights(rewardFunctions[i]);
        optimalPolicies.push_back(LSTDQ::lspi(lstdqDemonstrations, mutableMdp,
                                              false));
    }
    /*
     * This is not interesting because "true MDP" should be MDP_m
     *
    cout << "Calculating corresponding summed losses of policies"
         << " >>evaluated in the true MDP<<..." << endl;
    vector<double> optimalPoliciesRealSummedLoss(N_REWARD_FUNCTIONS);
    for (int i = 0; i < N_REWARD_FUNCTIONS; ++i)
    {
        // Evaluate in true MDP
        auto weights = LSTDQ::lstdq(lstdqDemonstrations, optimalPolicies[i],
                                    mdp);
        double loss = BMT::loss(weights, optWeights, uniqueStates, cmp, true);
        optimalPoliciesRealSummedLoss[i] = loss;
        cout << "\t(" << i << ")\t" << loss << endl;
    }
    */
    // Win rates
    // cout << "Calculating corresponding average win rates (vs. random)..."
    //      << endl;
    // vector<double> optimalWinRates;
    // for (auto pi : optimalPolicies)
    //     optimalWinRates.push_back(
    //         getAverageOptimalUtility(pi, mdp, 0, nWinRateDemonstrations, -1));
    // double programmaticOptimalScore
    //     = getAverageOptimalUtility(optimalPolicy, mdp, 0,
    //                                nWinRateDemonstrations, -1);
    // double randomScore
    //     = getAverageOptimalUtility(randomPolicy, mdp, 0,
    //                                nWinRateDemonstrations, -1);
    // double randomRandomScore
    //     = getAverageOptimalUtility(randomRandomPolicy, mdp, 0,
    //                                nWinRateDemonstrations, -1);
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
        // TODO: Think about this:
        // Sampled reward functions losses w.r.t. true reward functions
        // should here be a cap of how close we can get...
    }


    // Initializing our experts using programmatic optimum weights
    /* NOTE: Assume that the experts have an unknown reward function that gives
     * rise to these values. I.e. the optimal expert[0] has the same reward
     * function as the real game (terminal rewards for win/loss), and the other
     * experts may have a completely arbitrary reward function.
     */

    /*****************************************************************
    vector<SoftmaxPolicy> experts;
    // Expert 0 has only information about DOUBLETS
    experts.push_back(SoftmaxPolicy(&cmp, optWeights, EXPERT_TEMPERATURES[0]));
    for (int i = 0; i <= 12; ++i)
    {
        if (
                i != TicTacToeCMP::FEATURE_DOUBLETS_X
             && i != TicTacToeCMP::FEATURE_DOUBLETS_O
           )
        {
            double tmp = experts.back().getWeights()[i];
            experts.back().setWeight(i, 0.01 * tmp);
        }
    }

    // Expert 1 has only information about FORKS
    experts.push_back(SoftmaxPolicy(&cmp, optWeights, EXPERT_TEMPERATURES[1]));
    for (int i = 0; i <= 12; ++i)
    {
        if (
                i != TicTacToeCMP::FEATURE_FORKS_X
             && i != TicTacToeCMP::FEATURE_FORKS_O
           )
        {
            double tmp = experts.back().getWeights()[i];
            experts.back().setWeight(i, 0.001 * tmp);
        }
    }
    experts.back().setWeight(TicTacToeCMP::FEATURE_TRIPLETS_X, 0);

    experts.push_back(SoftmaxPolicy(&cmp, optWeights, EXPERT_TEMPERATURES[2]));
    for (int i = 0; i <= 12; ++i)
    {
        if (
                i != TicTacToeCMP::FEATURE_SINGLETS_X
             && i != TicTacToeCMP::FEATURE_SINGLETS_O
             && i != TicTacToeCMP::FEATURE_CROSSPOINTS_X
             && i != TicTacToeCMP::FEATURE_CROSSPOINTS_O
             && i != TicTacToeCMP::FEATURE_CORNERS_X
             && i != TicTacToeCMP::FEATURE_CORNERS_O
             && i != TicTacToeCMP::FEATURE_CENTER
             && i != TicTacToeCMP::FEATURE_TRIPLETS_O
           )
        {
            double tmp = experts.back().getWeights()[i];
            experts.back().setWeight(i, 0.01 * tmp);
        }
    }
    double tmp = experts.back().getWeights()[TicTacToeCMP::FEATURE_TRIPLETS_O];
    experts.back().setWeight(TicTacToeCMP::FEATURE_TRIPLETS_O, 0.035 * tmp);
    *****************************************/

    // Expert 0 misses information about Forks and Singlets
    // experts.back().setWeight(TicTacToeCMP::FEATURE_FORKS_X, 0);
    // experts.back().setWeight(TicTacToeCMP::FEATURE_FORKS_O, 0);
    // experts.back().setWeight(TicTacToeCMP::FEATURE_SINGLETS_X, 0);
    // experts.back().setWeight(TicTacToeCMP::FEATURE_SINGLETS_O, 0);
//                                    // Expert 1 misses information about Doublets
//                                    experts.push_back(SoftmaxPolicy(&cmp, optWeights, EXPERT_TEMPERATURES[1]));
//                                    experts.back().setWeight(TicTacToeCMP::FEATURE_DOUBLETS_X, 0);
//                                    experts.back().setWeight(TicTacToeCMP::FEATURE_DOUBLETS_O, 0);
//                                    /*
//                                        experts.push_back(SoftmaxPolicy(&cmp, optWeights, EXPERT_TEMPERATURES[1]));
//                                        double tmp;
//                                        tmp = experts.back().getWeights()[TicTacToeCMP::FEATURE_DOUBLETS_X];
//                                        experts.back().setWeight(TicTacToeCMP::FEATURE_DOUBLETS_X, 0.5 * tmp);
//                                        tmp = experts.back().getWeights()[TicTacToeCMP::FEATURE_DOUBLETS_O];
//                                        experts.back().setWeight(TicTacToeCMP::FEATURE_DOUBLETS_O, 0.5 * tmp);
//                                    */
//                                    // Expert 2 misses information about center and corners
//                                    experts.push_back(SoftmaxPolicy(&cmp, optWeights, EXPERT_TEMPERATURES[2]));
//                                    experts.back().setWeight(TicTacToeCMP::FEATURE_CENTER, 0);
//                                    experts.back().setWeight(TicTacToeCMP::FEATURE_CORNERS_X, 0);
//                                    experts.back().setWeight(TicTacToeCMP::FEATURE_CORNERS_O, 0);
//                                    // Expert 3 is noisy
//                                    experts.push_back(SoftmaxPolicy(&cmp, optWeights, EXPERT_TEMPERATURES[3]));
//                                    // // Expert 0 is optimal
//                                    // experts.push_back(SoftmaxPolicy(&cmp, optWeights, EXPERT_TEMPERATURES[0]));
//                                    // // Expert 2 misses information about Singlets
//                                    // experts.push_back(SoftmaxPolicy(&cmp, optWeights, EXPERT_TEMPERATURES[2]));
//                                    // experts.back().setWeight(TicTacToeCMP::FEATURE_SINGLETS_X, 0);
//                                    // experts.back().setWeight(TicTacToeCMP::FEATURE_SINGLETS_O, 0);


    /*
    // l_m per prior reward function
    // Take expert weights as true value function of MDP_m
    // Take reward function and use its optimal policy
    // Calculate the loss of this optimal policy in MDP_m
    cout << "Summed loss of reward-optimal policies w.r.t. true rho values"
         << " rho_m == MDP_m == V_m == (expert weights)" << endl;
    vector<vector<double>> expertRewardLoss(N_EXPERTS,
                                            vector<double>(N_REWARD_FUNCTIONS,
                                                           0));
    for (int i = 0; i < N_EXPERTS; ++i)
    {
        cout << "\t";
        for (int j = 0; j < N_REWARD_FUNCTIONS; ++j)
        {
            double loss = BMT::loss(experts[i].getWeights(),
                                    optimalPolicies[j].getWeights(),
                                    // uniqueStates, cmp, true);
                                    uniqueStates, cmp, false);
            cout << fixed << loss << "\t";
            expertRewardLoss[i][j] = loss;
        }
        cout << endl;
    }
    */

    /*
    // True expert losses w.r.t. true MDP
    cout << "Calculating summed losses for each expert w.r.t. true MDP (rf),"
         << " to be compared with \"final loss\" later:" << endl;
    for (int i = 0; i < N_EXPERTS; ++i)
    {
        // auto expertWeights = LSTDQ::lstdq(lstdqDemonstrations, experts[i], mdp);
        auto expertWeights = experts[i].getWeights();
        double loss = BMT::loss(expertWeights, optWeights, uniqueStates,
                                cmp, true); // sum?
        cout << "\t" << loss << endl;
    }
    */

    const int MAX_EXPERT_PLAYOUTS = N_N_EXPERT_PLAYOUTS[N_EXPERT_PLAYOUTS-1];
    vector<vector<Demonstration>> expertBaseDemonstrations;
    // vector<vector<Demonstration>> expertBmtLstdqDemonstrations;
    for (int i = 0; i < N_EXPERTS; ++i)
    {
        auto demonstrations
            = generateDemonstrations(mdp, {&experts[i]},
                                     -1, /* horizon = playout */
                                     //10000,
                                     MAX_EXPERT_PLAYOUTS,
                                     0, /* initialState */
                                     false); /* No print */
        expertBaseDemonstrations.push_back(demonstrations);
        // auto it = demonstrations.begin();
        // vector<Demonstration> bmtDemonstrations(it,
        //                                         it + (MAX_EXPERT_PLAYOUTS / 2));
        // expertBmtLstdqDemonstrations.push_back(bmtDemonstrations);
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
            // cout << demonstrations.size() << " == " << N_EXPERT_DEMONSTRATIONS
            //      << endl;
            assert(demonstrations.size() == N_EXPERT_DEMONSTRATIONS);

            // auto demonstrations
            //     = generateDemonstrations(mdp, {&experts[i]},
            //                              -1, /* horizon = playout */
            //                              N_EXPERT_DEMONSTRATIONS,
            //                              0, /* initialState */
            //                              false); /* No print */
            expertDemonstrations.push_back(demonstrations);
        }

        // TODO: Should perhaps be modified prior
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
        cout << "Sampling " << N_POLICY_SAMPLES << " policies from " << N_EXPERTS
             << " experts..." << endl;
        // cout << "TESTING WITH 0th POLICY = TRUE EXPERT POLICY" << endl;
        vector<vector<Policy*>> sampledPolicies(N_EXPERTS);
        for (int i = 0; i < N_EXPERTS; i++)
        {
            // sampledPolicies[i].push_back(&experts[i]);
            // for (int k = 0; k < N_POLICY_SAMPLES - 1; ++k)
            for (int k = 0; k < N_POLICY_SAMPLES; ++k)
                sampledPolicies[i].push_back(&posteriorPolicies[i].samplePolicy());
        }

        // Initialize BMT calculations for each expert
        cout << "Initializing " << N_EXPERTS << " BMT objects..." << endl;
        vector<BMT> bmts;
        for (int i = 0; i < N_EXPERTS; ++i)
        {
            // set<int> uniqueLossCalculationStates;
            // for (Demonstration demo : expertBaseDemonstrations[i])
            //     for (Transition tr : demo)
            //         uniqueLossCalculationStates.insert(tr.s);
            // Using expert demonstrations instead of lstdqDemonstrations since
            // this gives better support for the posterior policy.
            BMT bmt(mutableMdp,
                    // lstdqDemonstrations,
                    // expertBmtLstdqDemonstrations[i],
                    expertDemonstrations[i],
                    rewardFunctions,
                    optimalPolicies,
                    sampledPolicies[i],
                    EXPERT_OPTIMALITY_PRIORS[i],
                    useModel, // withModel
                    useSum
                    //,&uniqueLossCalculationStates
                    );
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

        // double pMax = -DBL_MAX;
        // int rMax = 0;
        // for (int i = 0; i < N_REWARD_FUNCTIONS; ++i)
        // {
        //     if (probabilityProducts[i] > pMax)
        //     {
        //         pMax = probabilityProducts[i];
        //         rMax = i;
        //     }
        // }

        if (true)
        {
            vector<double> magicMeanRewardFunction(cmp.nFeatures(), 0);
            // Printing of loss matrices
            for (int i = 0; i < N_EXPERTS; ++i)
            {
                BMT& bmt = bmts[i];
                int K = N_POLICY_SAMPLES;
                int N = N_REWARD_FUNCTIONS;
                if (true) // loss matrix
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
                // cout << "Reward function probabilities (unnormalised):" << endl;
                // for (int j = 0; j < N; ++j)
                //     cout << fixed << bmt.getRewardProbability(j) << "\t";
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
                // double meanLoss = 0;
                // rewardExpertLoss
                for (int j = 0; j < N_REWARD_FUNCTIONS; ++j)
                {
                    double p = rewardProbabilities[j];
                    // meanLoss += p * expertRewardLoss[i][j];
                    for (int k = 0; k < meanRewardFunction.size(); ++k)
                    {
                        meanRewardFunction[k] += p * rewardFunctions[j][k];
                        magicMeanRewardFunction[k] += p / ((double) N_EXPERTS)
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
            //  auto meanRFOptimalPolicy = LSTDQ::lspi(lstdqDemonstrations,
                auto meanRFOptimalPolicy = LSTDQ::lspi(expertDemonstrations[i],
                                                       mutableMdp);
                mutableMdp.setRewardWeights(trueRewardFunctions[i]); // update
                auto meanRFOptimalPolicyInTrueMDPWeights
                 // = LSTDQ::lstdq(lstdqDemonstrations, meanRFOptimalPolicy,
                    = LSTDQ::lstdq(expertDemonstrations[i], meanRFOptimalPolicy,
                                   mutableMdp); // mdp);
                auto MDP_m_optimalWeights =  // Update:
                    experts[i].getWeights(); //   expert[i] is OptimalPolicy
                double meanRFOptimalPolicyLoss
                    // = bmts[i].loss(meanRFOptimalPolicyInTrueMDPWeights,
                    // = BMT::loss(   meanRFOptimalPolicyInTrueMDPWeights,
                    = bmts[i].loss(meanRFOptimalPolicyInTrueMDPWeights,
                                   MDP_m_optimalWeights, // optWeights, // experts[i].getWeights(),
                                   // uniqueStates, // BMT::
                                   // cmp, // BMT ::
                                   useSum);
                double meanRFOptimalPolicyScore
                    = getAverageOptimalUtility(meanRFOptimalPolicy, mdp, 0,
                                               nWinRateDemonstrations, -1);
                // Compared to true ?MDP_m == ?rho_m == expert_m.weights
                // cout << "ARGMAX RHO(" << rMax << ") LOSS: "
                //      << expertRewardLoss[i][rMax] << endl;
                // cout << "MEAN LOSS: " << meanLoss << endl;
                cout << "\033[0;34mMEAN RF POLICY* LOSS IN MDP_m: "
                     << default_console
                     << meanRFOptimalPolicyLoss << endl;
                cout << "\033[0;34mMEAN RF POLICY* SCORE IN MDP: "
                     << default_console
                     << meanRFOptimalPolicyScore << endl;

                cout << endl;
            }

            /*
             * Magic mean printing, hopefully good
             */
            // Calculate ~optimal policy for magic mean
            mutableMdp.setRewardWeights(magicMeanRewardFunction);
            auto magicMeanRFOptimalPolicy = LSTDQ::lspi(lstdqDemonstrations,
                                                        mutableMdp);
            // // Evaluate magic ~optimal policy in true environment
            // auto magicMeanRFOptimalPolicyInTrueMDPWeights
            //     = LSTDQ::lstdq(lstdqDemonstrations, magicMeanRFOptimalPolicy,
            //                    mdp);
            // double magicMeanRFOptimalPolicyLoss
            //     = BMT::loss(magicMeanRFOptimalPolicyInTrueMDPWeights,
            //                 optWeights,
            //                 uniqueStates, cmp,
            //                 useSum); // true=L1 norm
            // cout << "~~~ MAGIC LOSS ~~~\t" << magicMeanRFOptimalPolicyLoss
            //      << endl;

            double magicMeanRFOptimalPolicyScore
                = getAverageOptimalUtility(magicMeanRFOptimalPolicy, mdp, 0,
                                           nWinRateDemonstrations, -1);
            cout << "\033[0;32m~~~ MAGIC SCORE ~~~\t"
                 << default_console
                 << magicMeanRFOptimalPolicyScore
                 << endl;

        }
    }
}

void test_BMT3()
{
    clock_t t = clock();
    srand((unsigned)time(NULL));

    /************************************************
    * * * * * * * * * Main parameters * * * * * * * *
    ************************************************/
    const int    N_EXPERT_DEMO_HORIZONS   =       11;
    const int    T_EXPERT_DEMO_HORIZONS[] =
   {1000, 1, 2, 10, 20, 50, 100, 150, 200, 500, 1000}; // Max 5000 makes sense
      // {2000}; // Max 5000 makes sense
    const int    N_EXPERT_DEMONSTRATIONS  =        1;
    const int    N_REWARD_FUNCTIONS       =        8;
    const int    N_POLICY_SAMPLES         =       30;
    const int    N_EXPERTS                =        5;
    // const double REWARD_ALPHA             =     0.05;
    const double EXPERT_TEMPERATURES[]    =
                    //{0.0001, 0.75, 1.00, 3.00, 5.00};
                      {0.01, 0.15, 0.20, 3.00, 3.50};
                    //{0.50, 0.75, 1.00, 3.00, 3.50};
                    //{0.10, 0.15, 0.20, 1.00, 1.50};
    const double EXPERT_OPTIMALITY_PRIORS[]    =
// /* ACTUALLY ALPHA IN GAMMA CDF */ {1, 1, 1, 5, 5};
                       //  {2.0, 2.0, 2.0, 0.5, 0.5};
                           {6.0, 6.0, 6.0, 6.0, 6.0};
    /************************************************
     ************************************************/
    /************************************************
    * * * * * * Semi static parameterrs * * * * * * *
    ************************************************/
    const bool PRINT_DEBUG               =      true;
    const bool useSum                    =      true;
    const bool useModel                  =     false;
    const int lstdDemonstrationLength    =       100;
    // const int N_Q_APPROXIMATION_PLAYOUTS =         1;
    // const int T_Q_APPROXIMATION_HORIZON  =        24; // since gamma = 0.75
    // const int T_Q_APPROXIMATION_HORIZON  =        10; // since gamma = 0.5
    /************************************************
     ************************************************/

    // MDP setup
    const double gamma = 0.75;
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

    // auto lstdqDemonstrations
    //     = generateRandomMDPDemonstrations(mdp, lstdDemonstrationLength);
    set<int> uniqueStates;
    for (Demonstration demo : lstdqDemonstrations)
        for (Transition tr : demo)
            uniqueStates.insert(tr.s);

    // Optimal policy
    cout << "Calculating optimal policy (LSPI)..." << endl;
    DeterministicPolicy lspiPolicy = LSTDQ::lspi(lstdqDemonstrations, mdp,
                                                 true, 1e-7, useModel);

    // TODO: Save/read on file?
    // Global set of reward functions
    // Find rewards with some noise, for more variance... (SoftmaxPolicy)
    cout << "Sampling " << N_REWARD_FUNCTIONS << " reward functions..." << endl;
    /*
    SoftmaxPolicy rewardPolicy(&cmp, lspiPolicy.getWeights(), 20.0);
    vector<vector<double>> rewardFunctions = 
        sampleRewardFunctions(N_REWARD_FUNCTIONS, N_Q_APPROXIMATION_PLAYOUTS,
                              T_Q_APPROXIMATION_HORIZON, lstdqDemonstrations,
                              rewardPolicy, mdp);
                           // lspiPolicy, mdp);
    */
    vector<vector<double>> rewardFunctions;
    for (int i = 0; i < N_REWARD_FUNCTIONS; ++i)
    {
        vector<double> rf(cmp.nFeatures(), 0);
        rf[i % cmp.nFeatures()] = 1;
        rewardFunctions.push_back(rf);
        continue;
        //test_BWT2_sampleRewardFunction(mdp.cmp->nFeatures());
        // auto rf = kernel.sample_multinomial();
        // assert(rf.size() == mdp.cmp->nFeatures());
        // rewardFunctions.push_back(rf);
        /*
         * Dirichlet sampling
         */
        /*
        {
            static gsl_rng *r_global = NULL;
            static const vector<double> alphas(cmp.nFeatures(),
                                               REWARD_ALPHA);
            vector<double> multinomial(cmp.nFeatures(), 0);
            if (r_global == NULL)
            {
                r_global = gsl_rng_alloc(gsl_rng_default);
                gsl_rng_set(r_global, (unsigned)time(NULL));
            }
            gsl_ran_dirichlet(r_global, cmp.nFeatures(), &alphas[0],
                              &multinomial[0]);
            // for (int f = 0; f < cmp.nFeatures(); ++f)
            //     rf[f] *= multinomial[f];
            rewardFunctions.push_back(multinomial);
        }
        */
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

    // TODO: Save/read on file?
    // Global set of corresponding optimal policies
    cout << "Creating " << N_REWARD_FUNCTIONS
         << " corresponding optimal policies..." << endl;
    vector<DeterministicPolicy> optimalPolicies;
    for (int i = 0; i < N_REWARD_FUNCTIONS; ++i)
    {
        mdp.setRewardWeights(rewardFunctions[i]);
        optimalPolicies.push_back(LSTDQ::lspi(lstdqDemonstrations, mdp, false,
                                              1e-7, useModel));
    }
    mdp.setRewardWeights(trueRewardFunction); // TODO: Probably not necessary
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
        // TODO: Think about this:
        // Sampled reward functions losses w.r.t. true reward functions
        // should here be a cap of how close we can get...
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
            // TODO: Perturb expertRf
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
            auto expertDeterministic = LSTDQ::lspi(lstdqDemonstrations, mdp);
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
        auto expertWeights = LSTDQ::lstdq(lstdqDemonstrations, experts[i], mdp);
        // lspiPolicy.getWeights() // Opt weights
        double loss = BMT::loss(expertWeights, optimalWeights, uniqueStates,
                                cmp, useSum); // sum?
        cout << "\t" << loss << endl;
    }

    const int MAX_EXPERT_HORIZON
        = T_EXPERT_DEMO_HORIZONS[N_EXPERT_DEMO_HORIZONS-1];
    vector<vector<Demonstration>> expertBaseDemonstrations;
    // vector<vector<Demonstration>> expertBmtLstdqDemonstrations;
    for (int i = 0; i < N_EXPERTS; ++i)
    {
        auto demonstrations
            = generateDemonstrations(mdp, {&experts[i]},
                                     MAX_EXPERT_HORIZON, /* horizon = playout */
                                     N_EXPERT_DEMONSTRATIONS, /* nDemonstrations */
                                     -1, /* initialState uniform random */
                                     false); /* No print */
        expertBaseDemonstrations.push_back(demonstrations);
        // auto it = demonstrations.begin();
        // vector<Demonstration> bmtDemonstrations(it,
        //                                         it + (MAX_EXPERT_PLAYOUTS / 2));
        // expertBmtLstdqDemonstrations.push_back(bmtDemonstrations);
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
            expertDemonstrations.push_back({demonstration});
            // auto demonstrations
            //     = generateDemonstrations(mdp, {&experts[i]},
            //                              T_EXPERT_DEMO_HORIZON,
            //                              N_EXPERT_DEMONSTRATIONS,
            //                              -1, /* initialState uniform random */
            //                              false); /* No print */
            // expertDemonstrations.push_back(demonstrations);
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
        if (true && PRINT_DEBUG)
        {
            cout.precision(numeric_limits< double >::digits10 - 12);
            cout << "\tSampled policies state action probabilities (first policy "
                 << "for each expert only:" << endl;
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

        // double pMax = -DBL_MAX;
        // int rMax = 0;
        // for (int i = 0; i < N_REWARD_FUNCTIONS; ++i)
        // {
        //     if (probabilityProducts[i] > pMax)
        //     {
        //         pMax = probabilityProducts[i];
        //         rMax = i;
        //     }
        // }

        // // Get best policy via argmax rho product
        // mdp.setRewardWeights(rewardFunctions[rMax]);
        // DeterministicPolicy bestPolicy = LSTDQ::lspi(lstdqDemonstrations, mdp);

        // // Evaluate it in the real MDP
        // mdp.setRewardWeights(trueRewardFunction);
        // vector<double> bestPolicyTrueMDPWeights
        //     = LSTDQ::lstdq(lstdqDemonstrations, bestPolicy, mdp);

        /*
        double finalLoss = BMT::loss(bestPolicyTrueMDPWeights, optimalWeights,
                                     uniqueStates, cmp, true); // sum
        cout << "Final loss, approx pi* from rho_" << rMax
             << " = argmax rho product evaluated in true MDP: " << endl << "\t"
             << finalLoss
             << endl;
        */

        if (true)
        {
            vector<double> magicMeanRewardFunction(cmp.nFeatures(), 0);
            // Printing of loss matrices
            for (int i = 0; i < N_EXPERTS; ++i)
            {
                BMT& bmt = bmts[i];
                int K = N_POLICY_SAMPLES;
                int N = N_REWARD_FUNCTIONS;
                if (true) // loss matrix
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
                // cout << "Reward function probabilities (unnormalised):" << endl;
                // for (int j = 0; j < N; ++j)
                //     cout << fixed << bmt.getRewardProbability(j) << "\t";
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

/**                           **/
/** BEGIN OF PASTED FROM BMT4 **/
/**                           **/
                // Average reward function
                vector<double> meanRewardFunction(cmp.nFeatures(), 0);
                // double meanLoss = 0;
                // rewardExpertLoss
                for (int j = 0; j < N_REWARD_FUNCTIONS; ++j)
                {
                    double p = rewardProbabilities[j];
                    // meanLoss += p * expertRewardLoss[i][j];
                    for (int k = 0; k < meanRewardFunction.size(); ++k)
                    {
                        meanRewardFunction[k] += p * rewardFunctions[j][k];
                        magicMeanRewardFunction[k] += p / ((double) N_EXPERTS)
                                                        * rewardFunctions[j][k];
                    }
                }
                cout << "Expected reward function:" << endl;
                for (auto r : meanRewardFunction)
                    cout << r << "\t";
                cout << endl;

                // Take expected reward function m --> get ~optimal policy m
                mdp.setRewardWeights(meanRewardFunction);
                auto meanRFOptimalPolicy = LSTDQ::lspi(lstdqDemonstrations, mdp,
                                                       true, 1e-7, useModel);
                // Evaluate ~optimal policy m in true environment
                mdp.setRewardWeights(trueRewardFunctions[i]);
                vector<double> MDP_m_optimalWeights;
                if (multipleTrueRewardFunctions)
                    MDP_m_optimalWeights = experts[i].getWeights();
                else
                    MDP_m_optimalWeights = optimalWeights;
                auto meanRFOptimalPolicyInTrueMDPWeights
                    = LSTDQ::lstdq(lstdqDemonstrations, meanRFOptimalPolicy,
                                   mdp);
                // Calculate loss w.r.t. true optimalWeights
                double meanRFOptimalPolicyLoss
                    // = BMT::loss(experts[i].getWeights(),
                    = bmts[i].loss(meanRFOptimalPolicyInTrueMDPWeights,
                                   MDP_m_optimalWeights, // optimalWeights
                                   // meanRFOptimalPolicy.getWeights(),
                                   useSum); // true=L1 norm
                                // cmp, true); // true=L1 norm
                                // cmp, false); // false=sup norm
// /**/         double meanRFOptimalPolicyScore
// /**/             = getAverageOptimalUtility(meanRFOptimalPolicy, mdp, 0,
// /**/                                        nWinRateDemonstrations, -1);
                // Compared to true ?MDP_m == ?rho_m == expert_m.weights
                // cout << "ARGMAX RHO(" << rMax << ") LOSS: "
                //      << expertRewardLoss[i][rMax] << endl;
                // cout << "MEAN LOSS: " << meanLoss << endl;
                cout << "MEAN RF POLICY* LOSS IN MDP_m: "
                     << meanRFOptimalPolicyLoss << endl;
// /**/         cout << "MEAN RF POLICY* SCORE IN MDP: "
// /**/              << meanRFOptimalPolicyScore << endl;
                cout << endl;
            }

            /*
             * Magic mean printing, hopefully good
             */
            // TODO: Print "magic loss" not only magic score
            // Calculate ~optimal policy for magic mean
            mdp.setRewardWeights(magicMeanRewardFunction);
            auto magicMeanRFOptimalPolicy = LSTDQ::lspi(lstdqDemonstrations,
                                                        mdp, true, 1e-7,
                                                        useModel);
            // Evaluate magic ~optimal policy in true environment
            mdp.setRewardWeights(trueRewardFunction);
            auto magicMeanRFOptimalPolicyInTrueMDPWeights
                = LSTDQ::lstdq(lstdqDemonstrations, magicMeanRFOptimalPolicy,
                               mdp);
            double magicMeanRFOptimalPolicyLoss
                = BMT::loss(// magicMeanRFOptimalPolicy.getWeights(),
                            magicMeanRFOptimalPolicyInTrueMDPWeights,
                            optimalWeights,
                            uniqueStates, cmp,
                            useSum); // true=L1 norm
            cout << "~~~ MAGIC LOSS ~~~\t" << magicMeanRFOptimalPolicyLoss
                 << endl;
// /**/     double magicMeanRFOptimalPolicyScore
// /**/         = getAverageOptimalUtility(magicMeanRFOptimalPolicy, mdp, 0,
// /**/                                    nWinRateDemonstrations, -1);
// /**/     cout << "~~~ MAGIC SCORE ~~~\t" << magicMeanRFOptimalPolicyScore
// /**/          << endl;
/**                           **/
/**   END OF PASTED FROM BMT4 **/
/**                           **/
        }
    }

    float seconds = ((float)(clock() - t))/CLOCKS_PER_SEC;
    cerr << "Finished in " << seconds << " seconds." << endl;
}

/*
double getAverageOptimalUtility(Policy& optimalPolicy, DiscreteMDP& mdp,
                                int state, int nPlayouts, int T);
double getExpectedOptimalReward(Policy& optimalPolicy, DiscreteMDP& mdp,
                                int state, int action, int nPlayouts, int T);
*/

/*
 * These methods only perform some playouts from an optimal policy to generate
 * natural noise to the optimal values.
 */
/*
double getExpectedOptimalReward(Policy& optimalPolicy, DiscreteMDP& mdp,
                                int state, int nPlayouts, int T)
{
    // Assuming optimal policy is stationary.
    int a = optimalPolicy.probabilities(state)[0].first;
    return getExpectedOptimalReward(optimalPolicy, mdp, state, a, nPlayouts, T);
}
*/

double getExpectedOptimalReward(Policy& optimalPolicy, DiscreteMDP& mdp,
                                int state, int action, int nPlayouts, int T)
{
    // // Assuming optimal policy is stationary.
    // int a = optimalPolicy.probabilities(state)[0].first;
    auto transitions = mdp.cmp->kernel->getTransitionProbabilities(state,
                                                                   action);

    // Q(s,a) where a = pi*(s)
    double Qsa = 0;

    double expectedVs2 = 0; // Expected V(s')
    for (auto tr : transitions)
    {
        int s2 = tr.first;
        double p = tr.second;
        double Qs2a = getAverageOptimalUtility(optimalPolicy, mdp, s2,
                                               transitions.size(),
                                               // transitions.size() * nPlayouts,
                                               T);
        double Qs2a2= getAverageOptimalUtility(optimalPolicy, mdp, s2,
                                               nPlayouts, T);
        Qsa += p * (mdp.getReward(s2) + mdp.gamma * Qs2a2);
        expectedVs2 += p * Qs2a;
    }

    return (Qsa - mdp.gamma * expectedVs2); // Average reward by definition
}

double getAverageOptimalUtility(Policy& optimalPolicy, DiscreteMDP& mdp,
                                int state, int nPlayouts, int T)
{
    vector<Demonstration> playouts
        = generateDemonstrations(mdp, {&optimalPolicy}, T, nPlayouts, state,
                                 false);
    // Calculate discounted sum of rewards
    double totalPayoff = 0;
    for (Demonstration d : playouts)
        for (int t = 0; t < d.size(); ++t)
            totalPayoff += d[t].r * pow(mdp.gamma, t); // starts with ^0
    return (totalPayoff / (double) nPlayouts);
}

vector<Demonstration> generateDemonstrations(DiscreteMDP& mdp,
                                             vector<Policy*> policies,
                                             int demonstrationLength,
                                             int nDemonstrations,
                                             int initialState,
                                             bool print)
{
    if (print)
    {
        cout << "Generating " << nDemonstrations << " demonstrations";
        if (demonstrationLength > -1)
            cout << " of length " << demonstrationLength;
        cout << "..." << endl;
    }

    vector<Demonstration> demonstrations(nDemonstrations);

    bool printTicTacToe = false;

    for (int d = 0; d < nDemonstrations; ++d)
    {
        int currentState;
        if (initialState == -1) // Uniform initial state dist
            currentState = rand() % mdp.cmp->states;
        else
            currentState = initialState;
        int previousState;

        for (int t = 0;
             (demonstrationLength < 0 || t < demonstrationLength)
                 && !mdp.cmp->isTerminal(currentState);
             ++t)
        {
            Policy& pi = *policies[t % policies.size()];
            int action = sample(pi.probabilities(currentState));
            previousState = currentState;
            auto transitionProbabilities = 
                mdp.cmp->kernel->getTransitionProbabilities(currentState,
                                                            action);
            currentState = sample(transitionProbabilities);

            if (printTicTacToe) // print
            {
                ((TicTacToeCMP*)mdp.cmp)->printState(
                    TicTacToeCMP::State(
                        ((TicTacToeCMP*)mdp.cmp)->size, currentState));
                cout << "isTerminal: " << mdp.cmp->isTerminal(currentState)
                     << endl;
                cout << "winner: " << 
                    ((TicTacToeCMP*)mdp.cmp)->winner(
                    TicTacToeCMP::State(
                        ((TicTacToeCMP*)mdp.cmp)->size, currentState))
                     << endl;
            }

            double reward = mdp.getReward(currentState);
            demonstrations[d].push_back(Transition(previousState, action,
                                                   currentState, reward));
        }
        if (printTicTacToe) // print
            cout << endl << "-------------------------------------------------"
                 << endl;
    }

    if (print)
        cout << "Done!" << endl;
    return demonstrations;
}

void test_generateTTT()
{
    TicTacToeTransitionKernel kernel = TicTacToeTransitionKernel(3);
    TicTacToeCMP cmp(&kernel);
    TicTacToeMDP mdp(&cmp, true); // True rewards
    RandomTTTPolicy policyRandom(&cmp);
    OptimalTTTPolicy policyOptimal(&cmp);
    // vector<Policy*> policies = {&policyRandom, &policyOptimal};
    vector<Policy*> policies = { &policyRandom };
    vector<Demonstration> demonstrations
        = generateDemonstrations(mdp, policies, -1, 10, 0);

    // vector<double> optWeights(cmp.nFeatures());
    auto optWeights = LSTDQ::lstdq(demonstrations, policyOptimal, mdp);
    cout.precision(numeric_limits<double>::digits10 - 12);
    cout << "Optimal weights X(s,d,t,x,c,f) O(s,d,t,x,c,f) : " << endl;
    for (double d : optWeights)
        cout << fixed << d << "\t";
    cout << endl;
    cout << "LSPI weights: " << endl;
    DeterministicPolicy lspiPolicy = LSTDQ::lspi(demonstrations, mdp);
    auto lspiWeights = lspiPolicy.getWeights();
    for (double d : lspiWeights)
        cout << fixed << d << "\t";
    cout << endl;

    const int maxPrints = 0;
    int prints = 0;
    for (Demonstration d : demonstrations)
    {
        if (++prints > maxPrints)
            break;
        /*
        for (Transition tr : d)
        {
            int s = tr.s;
            cmp.printState(TicTacToeCMP::State(cmp.size, s));
        }
        */
        int s = d.back().s;
        cmp.printState(TicTacToeCMP::State(cmp.size, s));
        auto phi = cmp.features(s);
        cout << "Value X(s,d,t,x,c,f) O(s,d,t,x,c,f) = " << endl << "\t";
        for (double f : phi)
            cout << fixed << f << "\t";
        cout << endl << " *\t";
        double q = std::inner_product(phi.begin(), phi.end(),
                                      optWeights.begin(), 0.0);
        for (double d : optWeights)
            cout << fixed << d << "\t";
        cout << endl;
        cout << " =\t" << q << endl;
    }
    // cout << "Global count: " << globalCount << endl;
    // cout << "Global count2: " << globalCount2 << endl;
}

vector<Demonstration> generateSoftmaxDemonstrations(RandomMDP& mdp,
                                                    int demonstrationLength,
                                                    SoftmaxPolicy& pi)
{
    const int nDemonstrations = 1;
    // const int demonstrationLength = 1000;
    vector<Demonstration> demonstrations(nDemonstrations,
                                         Demonstration(demonstrationLength,
                                                       Transition()));

    for (int d = 0; d < nDemonstrations; ++d)
    {
        int currentState = rand() % mdp.cmp->states; // Uniform initial state dist
        int previousState;

        for (int t = 0; t < demonstrationLength; ++t)
        {
            int action = sample(pi.probabilities(currentState));
            previousState = currentState;
            auto transitionProbabilities = 
                mdp.cmp->kernel->getTransitionProbabilities(currentState,
                                                            action);
            currentState = sample(transitionProbabilities);
            double reward = mdp.getReward(currentState); // Deterministic state rewards
            demonstrations[d][t] = Transition(previousState, action,
                                              currentState, reward);
        }
    }

    return demonstrations;
}


/*
void test_BMT2()
{
    const double gamma = 0.95;
    const int states = 6;
    const int actions = 10;
    RandomTransitionKernel kernel(states, actions);
    RandomCMP cmp(&kernel);
    RandomMDP mdp(&cmp, gamma); // True MDP

    // Length of demonstrations
    const int T = 10000;

    // Set R of reward functions to be used in comparison with all tasks
    const int N = 10;
    vector<vector<double>> rewardSamplesTmp(N);
    vector<vector<double>*> rewardSamples(N);
    for (int j = 0; j < N; ++j)
    {
        rewardSamplesTmp[j] = test_BWT2_sampleRewardFunction(cmp.nFeatures());
        rewardSamples[j] = &rewardSamplesTmp[j];
    }

    // Random demonstrations used for policy evaluation (LSTDQ&LSPI)
    auto lspiDemonstrations = generateRandomMDPDemonstrations(mdp);

    // True reward function of MDP
    vector<double> trueRewardFunction1(cmp.nFeatures(), 0);
    for (int s = 0; s < cmp.states; ++s)
        trueRewardFunction1[s] = mdp.getReward(s);

    // True optimal policy of MDP
    DeterministicPolicy policyLspiTrue1 = LSTDQ::lspi(lspiDemonstrations, mdp);

    // True policy of expert 1
    double expertTemperature1 = 0.0001;
    SoftmaxPolicy expertPolicy1(&cmp, policyLspiTrue1.getWeights(),
                                expertTemperature1);

    auto expertDemonstrations1 = generateSoftmaxDemonstrations(mdp, T,
                                                               expertPolicy1);

    cout << "Expert action probabilities:" << endl;
    if (true)
    {
        for (int s = 0; s < states; ++s)
        {
            auto pr = expertPolicy1.probabilities(s);
            cout << "\t{ ";
            for (auto p : pr)
            {
                cout << "(" << p.first << ", "
                     << (p.second < 0.001 ? 0 : p.second) << "), ";
            }
            cout << " }" << endl;
        }
    }



    //////////////////////////////////////////////////////////////////////////////////////////
    // Let the first reward function be the TRUE for comparison
    rewardSamples[0] = &trueRewardFunction1;
    // rewardSamplesTmp[1] = policyLspiTrue.getWeights();
    // rewardSamples[1] = &rewardSamplesTmp[1];
    //////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////
    cout << "True reward function of expert 1: " << endl << "\t";
    for (auto r : trueRewardFunction1)
    {
        cout << r << "\t";
    }
    cout << endl;

    cout << "Sampled reward functions: " << endl;
    for (auto rho : rewardSamples)
    {
        cout << "\t";
        for (double rs : *rho)
        {
            cout << rs << "\t";
        }
        cout << endl;
    }
    //////////////////////////////////////////////////////////////////////////////////////////

    SoftmaxDirichletPrior policyPrior(actions);
    DirichletPolicyPosterior policyPosterior(policyPrior,
                                             expertDemonstrations1);

    // Policy samples
    const int K = 10;
    vector<Policy*> policySamples1(K);
    for (int k = 0; k < K; ++k)
        policySamples1[k] = &policyPosterior.samplePolicy();
    //////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////  True policy  //////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////
    SoftmaxPolicy policyOptimal(&cmp, policyLspiTrue1.getWeights(), 0.0001);
    policySamples1[0] = &expertPolicy1;   // Expert policy (data)
    policySamples1[1] = &policyLspiTrue1; // Zero loss policy
    policySamples1[2] = &policyOptimal;   // --> 0 loss policy
    //////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////

    BMT bmt(mdp, lspiDemonstrations, rewardSamples, policySamples1);

    cout.precision(numeric_limits< double >::digits10 - 12);
    cout << "Loss matrix" << endl;
    {
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
    cout << "Reward function probabilities (unnormalised):" << endl;
    for (int j = 0; j < N; ++j)
    {
        cout << fixed << bmt.getRewardProbability(j) << "\t";
    }
    cout << endl;
}
*/

/*
void test_BMT()
{
    const double gamma = 0.5;
    const int states = 6;
    const int actions = 10;
    RandomTransitionKernel kernel(states, actions);
    RandomCMP cmp(&kernel);
    RandomMDP mdp(&cmp, gamma);
    auto demonstrations = generateRandomMDPDemonstrations(mdp);

    vector<double> rewardFunction1 = { mdp.getReward(0), mdp.getReward(1),
                                       mdp.getReward(2), mdp.getReward(3),
                                       mdp.getReward(4), mdp.getReward(5),
                                       0 };
    vector<double> rewardFunction2 = { 10 , 20, 30, 40, 50, 60, 0 } ; // Should be horrible
    vector<double> rewardFunction3 = { mdp.getReward(0), mdp.getReward(1),
                                       0,                mdp.getReward(3),
                                       mdp.getReward(4), 0,
                                       0 };

    auto rewardFunctions = {&rewardFunction1, &rewardFunction2, &rewardFunction3};
    for (auto rho : rewardFunctions)
        normalize(*rho);

    const int N = rewardFunctions.size();

    // TODO: Use Dirichlet Policy Posterior to get reasonable policies.
    // Now have 1 bad and 1 good
    ConstPolicy policy1 = ConstPolicy(vector<int>(states, 0)); // Always action 0
    DeterministicPolicy policy2 = LSTDQ::lspi(demonstrations, mdp);
    ConstPolicy policy3 = ConstPolicy(vector<int>(states, 1)); // Always action 1
    // Softmax Policies based on optimal policy but with different temperatures
    SoftmaxPolicy policy4(&cmp, policy2.getWeights(), 0.00001);
    SoftmaxPolicy policy5(&cmp, policy2.getWeights(), 0.001);
    SoftmaxPolicy policy6(&cmp, policy2.getWeights(), 0.01);
    SoftmaxPolicy policy7(&cmp, policy2.getWeights(), 0.1);
    SoftmaxPolicy policy8(&cmp, policy2.getWeights(), 1.0);

    vector<Policy*> policies = {&policy1, &policy2, &policy3,
                                &policy4, &policy5, &policy6,
                                &policy7, &policy8};
    const int K = policies.size();

    // Debug print policy action
    for (int i = 0; i < K; ++i)
    {
        cout << "Policy " << i << " actions:\t";
        for (int s = 0; s < states; ++s)
            cout << "(" << policies[i]->probabilities(s)[0].first
                 << ", " << policies[i]->probabilities(s)[0].second << ") ";
        cout << endl;
    }

    BMT bmt(mdp, demonstrations, rewardFunctions, policies);

    if (false)
    {
        for (int s = 0; s < states; ++s)
        {
            auto pr = policy4.probabilities(s);
            cout << "\t{";
            for (auto p : pr)
            {
                cout << "(" << p.first << ", " << p.second << ")";
            }
            cout << "}" << endl;
        }
    }

    if (false)
    {
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        vector<int> sstates(demonstrations.size());
        for (Demonstration demo : demonstrations)
            for (Transition tr : demo)
                sstates.push_back(tr.s); // TODO: Alot of repetition of states here. Make it std::set
        cout << "Loss progression: ";
        for (double c = 0.1; c < 1; c += 0.1)
        {
            SoftmaxPolicy sp(&cmp, policy2.getWeights(), c);
            auto weights = LSTDQ::lstdq(demonstrations, sp, mdp);
            cout << c << ": " << bmt.loss(weights, policy2.getWeights(), sstates, cmp) << " ";
            // cout << endl;
            // for (int s = 0; s < states; ++s)
            // {
            //     auto pr = sp.probabilities(s);
            //     cout << "\t{";
            //     for (auto p : pr)
            //     {
            //         cout << "(" << p.first << ", " << p.second << ")";
            //     }
            //     cout << "}" << endl;
            // }
        }
        cout << endl;
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    }

    cout.precision(numeric_limits< double >::digits10 - 12);
    cout << "Loss matrix" << endl;
    {
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
    cout << "Reward function probabilities (unnormalised):" << endl;
    for (int j = 0; j < N; ++j)
    {
        cout << fixed << bmt.getRewardProbability(j) << "\t";
    }
    cout << endl;
}
*/

void test_dirichletPolicyPosterior()
{
    const double gamma = 0.5;
    const int states = 6;
    const int actions = 10;
    RandomTransitionKernel kernel(states, actions);
    RandomCMP cmp(&kernel);
    RandomMDP mdp(&cmp, gamma);
    auto demonstrations = generateRandomMDPDemonstrations(mdp, 500);

    SoftmaxDirichletPrior prior(actions);
    DirichletPolicyPosterior foo(prior, demonstrations);

    vector<int>& r5 = foo.getActionCounts(5);
    r5[5] = 555;
    vector<int>& r2 = foo.getActionCounts(2);
    r2[2] = 222;

    cout << "r5:\t";
    for (int c : r5)
        cout << c << " ";
    cout << endl;
    cout << "r2:\t";
    for (int c : r2)
        cout << c << " ";
    cout << endl;
    cout << "r2 ?:\t";
    for (int c : r2)
        cout << c << " ";
    cout << endl;

    vector<int> tests = {2,5,0};
    for (int test : tests)
    {
        for (int p = 0; p < 2; ++p)
        {
            Policy& pi = foo.samplePolicy();
            vector<pair<int,double>> ssmn;
            for (int i = 0; i < 2; ++i)
            {
                ssmn = pi.probabilities(test);
                cout << "\ts" << test << "mn(" << i << ") policy " << p << "\t";
                for (auto c : ssmn)
                    cout << c.second << " ";
                cout << endl;
            }
        }
    }


    // foo.samplePolicy();
    // foo.samplePolicy();
    return;
}

void normalize(vector<double>& v)
{
    double sum = 0;
    int k = v.size();
    for (int i = 0; i < k; ++i)
        sum += v[i];
    for (int i = 0; i < k; ++i)
        v[i] /= sum;
}

void test_lstdq_optpolicy()
{
    TicTacToeTransitionKernel cmpKernel = TicTacToeTransitionKernel(3);
    TicTacToeCMP tttCmp(&cmpKernel);
    RandomTTTPolicy randomPolicy(&tttCmp);
    OptimalTTTPolicy optimalPolicy(&tttCmp);

    const int n = 10000; // O(n)
    const int k = tttCmp.features().size();
    const double gamma = 1.0;

    vector<double> phi(k*n);
    vector<double> td(n*k);
    vector<double> b(k);

    int player = 2;
    for (int i = 0; i < n; ++i)
    {
        if (tttCmp.winner() != 0)
        {
            tttCmp.resetState();
            // player = 2; 
        }

        // Play 2 moves (X + O)
        bool terminal = false;

        do 
        {
            tttCmp.move(randomPolicy.action(tttCmp.currentState), player);
            player = (player == 2) ? 1 : 2;
            if (tttCmp.winner())
                terminal = true;
        } while (!terminal && player == 2);
        /*
        for (int i = 0; i < 2; ++i)
        {
            player = (player == 2) ? 1 : 2;
            // if (player == 2)
                tttCmp.move(randomPolicy.action(tttCmp.currentState), player);
            // else
            //     tttCmp.move(optimalPolicy.action(tttCmp.currentState), player);
            if (tttCmp.winner())
            {
                terminal = true;
                break;
            }
        }
        */

        TicTacToeCMP::State s2(tttCmp.currentState);

        auto features = tttCmp.features(s2.getState());
        // features[TicTacToeCMP::FEATURE_FORKS_X] = 0;
        // features[TicTacToeCMP::FEATURE_FORKS_O] = 0;
        // normalize(features);

        if (!terminal)
            s2.move(optimalPolicy.action(s2), 1);
        auto featuresPi = tttCmp.features(s2.getState());
        // featuresPi[TicTacToeCMP::FEATURE_FORKS_X] = 0;
        // featuresPi[TicTacToeCMP::FEATURE_FORKS_O] = 0;
        // normalize(featuresPi);

        double r;
        const double rewards[] = {0, 1, -1000, 0};
        int win = tttCmp.winner();
        r = rewards[win];
        for (int j = 0; j < k; ++j)
        {
            // phi[i * k + j] = features[j];
            phi[j * n + i] = features[j];
            if (terminal)
                td[i * k + j] = features[j];
            else
                td[i * k + j] = (features[j] - gamma * featuresPi[j]);
            if (r != 0)
                b[j] += r * features[j];
        }
    }

    cout << "..." << endl;

    vector<double> w = LSTDQ::solve(k, n, phi, td, b);

    cout << "LSTDQ(optimal policy) w* = ";
    for (int i = 0; i < k; ++i)
        cout << w[i] << " ";
    cout << endl;

    tttCmp.resetState();
    tttCmp.move(2,2, 1);
    tttCmp.move(2,1, 1);
    tttCmp.move(1,2, 1);
    test_tictactoecmp_print(tttCmp);
    double sum = 0;
    auto features = tttCmp.features(); 
    for (int i = 0; i < k; ++i)
        sum += features[i]*w[i];
    cout << "Value: " << sum << endl;

    wGlobal = w;
}

void test_tictactoetransitionkernel()
{
    cout << "*** Testing TicTacToeTransitionKernel..." << endl;
    TicTacToeTransitionKernel cmpKernel = TicTacToeTransitionKernel(3);
    DiscreteCMP cmp(&cmpKernel);
    // Occupied: 5, 7, 3, 0
    // ==> Valid actions: 1,2,4,6,8
    int s = pow(3,5)*2 + pow(3,7)*2 + pow(3,3)*1 + pow(3,0)*1;
    set<action> validActions = cmp.kernel->getValidActions(s);
    cout << "Occupied: 0 3 5 7. Valid actions: ";
    for (action a : validActions)
        cout << a << " ";
     cout << endl;
}

void test_tictactoecmp_print(TicTacToeCMP& tttCmp)
{
    tttCmp.printState();

    OptimalTTTPolicy policy(&tttCmp);

    set<action> validActions =
        tttCmp.kernel->getValidActions(tttCmp.currentState.getState());
    cout << "Valid actions: ";
    for (action a : validActions)
        cout << a << " ";
    if (validActions.size() > 0)
        cout << " (" << policy.action(tttCmp.currentState) << " optimal)";
    cout << endl;

    cout << "Features X(s,d,t,x,c,f) O(s,d,t,x,c,f) Raw(1-9): ";
    vector<double> w = tttCmp.features();

    for (double wi : w)
        cout << wi << " ";
    cout << endl;
}

void test_tictactoecmp()
{
    cout << "*** Testing TicTacToeCMP..." << endl;
    TicTacToeTransitionKernel cmpKernel = TicTacToeTransitionKernel(3);
    TicTacToeCMP tttCmp(&cmpKernel); //

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 2; ++j)
            tttCmp.move(i,j, 1);
    tttCmp.move(0,2, 2);
    tttCmp.move(1,2, 2);
    tttCmp.move(2,2, 2);
    test_tictactoecmp_print(tttCmp);


    tttCmp.resetState();
    tttCmp.move(1,2, 1);
    tttCmp.move(1,1, 2);
    test_tictactoecmp_print(tttCmp);
    tttCmp.move(0,1, 1);
    tttCmp.move(0,2, 2);
    tttCmp.move(0,0, 1);
    test_tictactoecmp_print(tttCmp);
    tttCmp.move(2,0, 1);
    tttCmp.move(1,0, 2);
    tttCmp.move(2,2, 1);
    test_tictactoecmp_print(tttCmp);

    tttCmp.resetState();
    test_tictactoecmp_print(tttCmp);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            tttCmp.move(i,j, 1);
    tttCmp.move(2,1, 2);
    test_tictactoecmp_print(tttCmp);

    tttCmp.resetState();
    tttCmp.move(1,0, 1);
    tttCmp.move(0,2, 1);
    test_tictactoecmp_print(tttCmp);

    // Fork 1
    tttCmp.resetState();
    tttCmp.move(2,2, 1);
    tttCmp.move(2,1, 1);
    tttCmp.move(1,2, 1);
    test_tictactoecmp_print(tttCmp);

    // Fork 2
    tttCmp.resetState();
    tttCmp.move(0,0, 1);
    tttCmp.move(0,2, 1);
    tttCmp.move(2,0, 1);
    test_tictactoecmp_print(tttCmp);

    // Fork 3
    tttCmp.resetState();
    tttCmp.move(2,0, 1);
    tttCmp.move(1,1, 1);
    tttCmp.move(1,2, 1);
    tttCmp.move(2,2, 1);
    test_tictactoecmp_print(tttCmp);

    // Fork 4
    tttCmp.resetState();
    tttCmp.move(2,0, 1);
    tttCmp.move(1,1, 1);
    tttCmp.move(1,2, 1);
    tttCmp.move(2,2, 1);
    tttCmp.move(0,2, 2);
    test_tictactoecmp_print(tttCmp);
    std::pair<double,double> results = optimal_vs_random();
    cout << "Optimal vs. random: X: " << results.first << " O: " << results.second << endl;
}

std::pair<double, double> optimal_vs_random()
{
    TicTacToeTransitionKernel cmpKernel = TicTacToeTransitionKernel(3);
    TicTacToeCMP tttCmp(&cmpKernel);
    OptimalTTTPolicy p1(&tttCmp);
    RandomTTTPolicy p2(&tttCmp);
    int p1wins = 0;
    int p2wins = 0;
    // auto winner = [&tttCmp]() mutable
    // {
    //     auto features = tttCmp.features();
    //     if (features[TicTacToeCMP::FEATURE_TRIPLETS_X] > 0)
    //         return 1;
    //     else if (features[TicTacToeCMP::FEATURE_TRIPLETS_O] > 0)
    //         return 2;
    //     else if (tttCmp.kernel->getValidActions(tttCmp.currentState.getState()).size() == 0)
    //         return 3;
    //     else
    //         return 0;
    // };

    auto checkWin = [&p1wins, &p2wins, &tttCmp]() mutable
    {
        int win = tttCmp.winner();
        if (win != 0)
        {
            tttCmp.resetState();
            switch (win)
            {
                case 1: ++p1wins; break;
                case 2: ++p2wins; break;
            }
        }
    };

    int i = 0;
    for (i = 0; i < 10000; ++i)
    {
        tttCmp.move( p1.action(tttCmp.currentState) , 1 );
        int win = tttCmp.winner();
        checkWin();
        if (win != 0)
            continue; // Always X first
        tttCmp.move( p2.action(tttCmp.currentState) , 2 );
        checkWin();
    }

    return std::pair<double,double> ( ((double) p1wins) / i , ((double) p2wins) / i );
}

void test_randomPolicy()
{
    RandomTTTData::fun();
}

void play_optimalTTTpolicy()
{
    TicTacToeTransitionKernel cmpKernel = TicTacToeTransitionKernel(3);
    TicTacToeCMP tttCmp(&cmpKernel);
    // OptimalTTTPolicy policy(&tttCmp);
    RandomTTTPolicy policy(&tttCmp);

    int moves = 0;
    while (1)
    {
        auto checkWin = [&tttCmp, &moves]() mutable
        {
            auto features = tttCmp.features();
            int win = 0;
            if (features[TicTacToeCMP::FEATURE_TRIPLETS_X] > 0)
                win = 1;
            else if (features[TicTacToeCMP::FEATURE_TRIPLETS_O] > 0)
                win = 2;

            ++moves;
            if (win || moves == 9)
            {
                if (win)
                    cout << " *** PLAYER " << (win == 1 ? "X" : "O") << " WINS *** " << endl;
                else
                    cout << " *** TIE *** " << endl;
                test_tictactoecmp_print(tttCmp);
                tttCmp.resetState();
                moves = 0;
            }
        };

        tttCmp.move( policy.action(tttCmp.currentState) , 1);
        checkWin();

        test_tictactoecmp_print(tttCmp);

        double sum = 0;
        auto features = tttCmp.features(); 
        int k = features.size();
        for (int i = 0; i < k; ++i)
            sum += features[i]*wGlobal[i];
        cout << "Value for X: " << sum << endl;

        int i,j;
        cin >> i >> j;

        tttCmp.move(i-1, j-1, 2);
        checkWin();
    }
}

int sample(vector<pair<int, double>> transitionProbabilities)
{
    double rr = r(); // [0,1]
    double sum = 0;
    for (auto p : transitionProbabilities)
    {
        int s2 = p.first;
        double prob = p.second;
        sum += prob;
        if (rr <= sum) // This assumes that the probabilities sum to 1
            return s2;
    }
    if (sum < 1-1e-6)
    {
        std::ostringstream ss;
        ss << "sum = " << sum << " < 1";
        throw std::runtime_error(ss.str());
    }
    else
    {
        return transitionProbabilities.back().first;
    }
}


vector<Demonstration> generateRandomMDPDemonstrations(RandomMDP& mdp,
                                                      int demonstrationLength)
{
    const int nDemonstrations = 1;
    // const int demonstrationLength = 500;
    vector<Demonstration> demonstrations(nDemonstrations,
                                         Demonstration(demonstrationLength, Transition()));

    for (int d = 0; d < nDemonstrations; ++d)
    {
        int currentState = rand() % mdp.cmp->states; // Uniform initial state dist
        int previousState;

        for (int t = 0; t < demonstrationLength; ++t)
        {
            int action = rand() % mdp.cmp->actions; // Completely random actions
            previousState = currentState;
            auto transitionProbabilities = 
                mdp.cmp->kernel->getTransitionProbabilities(currentState,
                                                            action);
            bool print = false;
            if (print && transitionProbabilities.size() > 1)
                cout << transitionProbabilities[0].first << " "
                     << transitionProbabilities[0].second << " "
                     << transitionProbabilities[1].first << " "
                     << transitionProbabilities[1].second;
            currentState = sample(transitionProbabilities);
            if (print && transitionProbabilities.size() > 1)
                cout << "to: " << currentState << endl;
            double reward = mdp.getReward(currentState); // Deterministic state rewards
            demonstrations[d][t] = Transition(previousState, action,
                                              currentState, reward);
        }
    }

    return demonstrations;
}

vector<double> test_lstdq_randomMDP(RandomMDP& mdp, Policy& pi)
{
    // cout << "*** Testing LSTDQ on RandomMDP" << endl;

    auto demonstrations = generateRandomMDPDemonstrations(mdp, 500);
    const int k = mdp.cmp->nFeatures();

    vector<double> w = LSTDQ::lstdq(demonstrations, pi, mdp);
    cout << "LSTDQ w* ( ";
    for (int s = 0; s < mdp.cmp->states; ++s)
    {
        cout << pi.probabilities(s)[0].first << " ";
    }
    cout << ") = ";
    for (int i = 0; i < k; ++i)
        cout << w[i] << " ";
    cout << endl;

    return w;
}

vector<vector<double>> sampleRewardFunctions(int nFunctions, int nPlayouts,
                                             int playoutHorizon,
                                             vector<Demonstration>&
                                                 lstdqDemonstrations,
                                             Policy& optimalPolicy,
                                             DiscreteMDP& mdp)
{
    vector<vector<double>> rewardFunctions;

    for (int i = 0; i < nFunctions; ++i)
    {
        vector<double> Phi;
        vector<double> avgRewards;
        for (Demonstration d : lstdqDemonstrations)
        {
            for (Transition tr : d)
            {
                double avgReward = getExpectedOptimalReward(optimalPolicy, mdp,
                                                            tr.s, tr.a, nPlayouts,
                                                            playoutHorizon);
                vector<double> features = mdp.cmp->features(tr.s, tr.a);
                copy(features.begin(), features.end(), back_inserter(Phi));
                avgRewards.push_back(avgReward);
            }
        }

        auto rewardFunction = BMT::solve_rect(Phi, avgRewards);
        rewardFunctions.push_back(rewardFunction);
    }

    return rewardFunctions;
}

void test_valueiteration()
{
    srand((unsigned)time(NULL));
    cout << "*** Testing ValueIteration on RandomMDP" << endl;

    const double gamma = 0.5;
    const int states = 3;
    const int actions = 2;
    RandomTransitionKernel kernel(states, actions);
    RandomCMP cmp(&kernel);
    RandomMDP mdp(&cmp, gamma);

    kernel.setTransitionProbability(0, 0, 0, 0.5);
    kernel.setTransitionProbability(0, 0, 1, 0.5);
    kernel.setTransitionProbability(0, 0, 2, 0.0);

    kernel.setTransitionProbability(1, 0, 0, 0.0);
    kernel.setTransitionProbability(1, 0, 1, 1.0);
    kernel.setTransitionProbability(1, 0, 2, 0.0);

    kernel.setTransitionProbability(2, 0, 0, 1.0);
    kernel.setTransitionProbability(2, 0, 1, 0.0);
    kernel.setTransitionProbability(2, 0, 2, 0.0);


    kernel.setTransitionProbability(0, 1, 0, 1.0);
    kernel.setTransitionProbability(0, 1, 1, 0.0);
    kernel.setTransitionProbability(0, 1, 2, 0.0);

    kernel.setTransitionProbability(1, 1, 0, 0.5);
    kernel.setTransitionProbability(1, 1, 1, 0.0);
    kernel.setTransitionProbability(1, 1, 2, 0.5);

    kernel.setTransitionProbability(2, 1, 0, 0.5);
    kernel.setTransitionProbability(2, 1, 1, 0.0);
    kernel.setTransitionProbability(2, 1, 2, 0.5);

    mdp.setRewardWeights({-5, 1, 10});


    if (false)
    {
        const double gamma = 0.5;
        const int states = 4;
        const int actions = 2;
        RandomTransitionKernel kernel(states, actions);
        RandomCMP cmp(&kernel);
        RandomMDP mdp(&cmp, gamma);

        if (false)
            for (int s = 0; s < states; ++s)
                mdp.setReward(s, mdp.getReward(s)*100);

        if (false)
        {
            // EX1
            for (int s = 0; s < states; ++s)
                for (int s2 = 0; s2 < states; ++s2)
                    for (int a = 0; a < actions; ++a)
                        kernel.setTransitionProbability(s, a, s2, 0);
            kernel.setTransitionProbability(0, 0, 1, 1);
            kernel.setTransitionProbability(1, 0, 0, 1);
            if (false)
                kernel.setTransitionProbability(2, 0, 2, 1);
            else
            {
                kernel.setTransitionProbability(2, 0, 2, 0.50);
                kernel.setTransitionProbability(2, 0, 1, 0.50);
            }
            kernel.setTransitionProbability(3, 0, 2, 1);
            kernel.setTransitionProbability(0, 1, 0, 1);
            kernel.setTransitionProbability(1, 1, 2, 1);
            kernel.setTransitionProbability(2, 1, 3, 1);
            kernel.setTransitionProbability(3, 1, 3, 1);
            mdp.setReward(0, -1);
            mdp.setReward(1, 0);
            mdp.setReward(2, 2);
            mdp.setReward(3, 1);
        }
        if (false)
        {
            for (int a = 0; a < actions; ++a)
                for (int s = 0; s < states; ++s)
                {
                    auto pp = kernel.getTransitionProbabilities(s, a);
                    for (auto ppp : pp)
                    {
                        cout << "Action " << a << " in state " << s
                             << " takes you to state " << ppp.first
                             << " with probability " << ppp.second << endl;
                    }
                }
        }

        if (false && true && true)
        {
            vector<double> phi = cmp.features(2, 0);
            cout << "phi(s=2, a=0) = ";
            for (auto f : phi)
            {
                cout << f << " ";
            }
            cout << endl;
            auto tp = cmp.kernel->getTransitionProbabilities(2, 0);
            for (auto p : tp)
            {
                cout << "P(s'=" << p.first << " | s=2, a=0) = " << p.second << "\t";
            }
            cout << endl;
            return;
        }
    }

    ValueIteration vi(&mdp);
    vi.computeStateActionValues();

    cout.precision(numeric_limits< double >::digits10 - 12);
    for (int a = 0; a < actions; ++a)
    {
        cout << "A" << a << ":\t";
        for (int s = 0; s < states; ++s)
            cout << fixed << vi.Q[s][a] << " | \t";
        cout << endl;
    }

    cout << "V" << ":\t";
    for (int s = 0; s < states; ++s)
    {
        double qMax = -DBL_MAX;
        int aMax;
        for (int a = 0; a < actions; ++a)
        {
            double q = vi.Q[s][a];
            if (q > qMax)
            {
                qMax = q;
                aMax = a;
            }
        }
        cout << fixed << qMax << "/A" << aMax << "  \t";
    }
    cout << endl;

    cout << "R" << ":\t";
    for (int s = 0; s < states; ++s)
        cout << fixed << mdp.getReward(s) << " | \t";
    cout << endl;

    // Print transition probs for each state and action
    cout << "Transition probabilities from states given actions:" << endl;
    for (int s = 0; s < states; ++s)
    {
        cout << "\tS" << s << ":" << endl;
        for (int a = 0; a < actions; ++a)
        {
            cout << "\t\tA" << a << ":\t";
            for (int s2 = 0; s2 < states; ++s2)
            {
                cout << kernel.getTransitionProbability(s, a, s2) << "\t";
            }
            cout << endl;
        }
    }

    /*
    // Print max transition prob for each state and action
    for (int a = 0; a < actions; ++a)
    {
        cout << "max s' { P(S=., A=" << a << ", S'=s') } = " << endl;
        cout << "\t";
        for (int s = 0; s < states; ++s)
        {
            double maxProb = 0;
            int maxState;
            for (int s2 = 0; s2 < states; ++s2)
            {
                double prob = kernel.getTransitionProbability(s, a, s2);
                if (prob > maxProb)
                {
                    maxProb = prob;
                    maxState = s2;
                }
            }
            cout << "S" << maxState;
            cout << fixed << " (" << maxProb << ")" << " \t";
        }
        cout << endl;
    }
    */

    vector<int> bestActions(states, 0);
    for (int s = 0; s < states; ++s)
    {
        int aMax;
        double qMax=-DBL_MAX;
        for (int a = 0; a < actions; ++a)
        {
            double q = vi.Q[s][a];
            if (q > qMax)
            {
                qMax = q;
                aMax = a;
            }
        }
        bestActions[s] = aMax;
    }

    ConstPolicy piOpt(bestActions);
    ConstPolicy pi00(0, states);
    ConstPolicy pi1(1, states);

    test_lstdq_randomMDP(mdp, pi00);
    test_lstdq_randomMDP(mdp, pi1);
    vector<double> w = test_lstdq_randomMDP(mdp, piOpt);

    /*
     * LSPI
     */
    auto demonstrations = generateRandomMDPDemonstrations(mdp, 500);
    DeterministicPolicy lspiPolicy = LSTDQ::lspi(demonstrations, mdp);
    cout << "LSPI w* ( ";
    for (int s = 0; s < mdp.cmp->states; ++s)
    {
        cout << lspiPolicy.probabilities(s)[0].first << " ";
    }
    cout << ") = ";
    for (int i = 0; i < mdp.cmp->nFeatures(); ++i)
        cout << lspiPolicy.getWeights()[i] << " ";
    cout << endl;
    /*
     */

    cout << "{V}:\t";
    for (int s = 0; s < states; ++s)
    {
        double avgValue = getAverageOptimalUtility(lspiPolicy, mdp, s, 100,
                                                   100 /* -1 for ttt */);
        cout << fixed << avgValue << "  \t\t";
    }
    cout << endl;

    cout << "~Q* - Q*" << endl;
    cout << "S\\A\t";
    for (int a = 0; a < actions; ++a)
        cout << a << "\t";
    cout << endl;
    for (int s = 0; s < states; ++s)
    {
        cout << s << "\t";
        for (int a = 0; a < actions; ++a)
        {
            vector<double> phi = cmp.features(s, a);
            double q = std::inner_product(phi.begin(), phi.end(), w.begin(), 0.0);
            cout << q - vi.Q[s][a] << "\t";
        }
        cout << endl;
    }

    int n = states * actions;
    int k = cmp.nFeatures();
    vector<double> A(n*k);
    vector<double> b(n);
    int i = 0;
    int j = 0;
    for (int s = 0; s < states; ++s)
    {
        for (int a = 0; a < actions; ++a)
        {
            auto tp = cmp.kernel->getTransitionProbabilities(s, a);
            double avgValue = 0;
            double avgReward = 0;
            for (auto tr : tp)
            {
                int s2 = tr.first;
                double p = tr.second;
                double v = -DBL_MAX;
                for (int a2 = 0; a2 < actions; ++a2)
                {
                    double q = vi.Q[s2][a2];
                    if (q > v)
                        v = q;
                }
                avgValue +=  p * v;
                avgReward += p * mdp.getReward(s2);
            }
            double approxAvgReward = vi.Q[s][a] - gamma * avgValue;
            double sampledAvgReward = getExpectedOptimalReward(lspiPolicy,
                                                               mdp, s, a,
                                                               5, 100);
            cout << "\tApproximate avg reward\t~<Q(" << s << "," << a << ")> =\t"
                 << approxAvgReward << endl;
            cout << "\tTrue avg reward\t\t <Q(" << s << "," << a << ")> =\t" << avgReward << endl;
            cout << "\tSampled avg reward\t {Q(" << s << "," << a << ")} =\t"
                 << sampledAvgReward << endl;

            vector<double> phi = cmp.features(s,a);
            for (double f : phi)
                A[i++] = f;
            // b[j++] = approxAvgReward;
            b[j++] = sampledAvgReward;
        }
    }

    if (false) // Print Ax=b system
    {
        i = 0;
        j = 0;
        for (double ai : A)
        {
            cout << ai << " ";
            if ((++i % k) == 0)
                cout << "\t" << b[j++] << endl;
        }
    }

    // auto rewards = BMT::solve_rect(A, b);

    cout << "Solved rewards from linear system generated using features &" 
         << " sampled average rewards:" << endl;
    auto rewardFunctions = sampleRewardFunctions(5, 1, 50, demonstrations,
                                                 lspiPolicy, mdp);
    for (auto rewardFunction : rewardFunctions)
    {
        for (auto r : rewardFunction)
            cout << r << "\t";
        cout << endl;
    }

    // Mathematica output
    /*
    bool first = true;
    cout << "{";
    for (int s = 0; s < states; ++s)
    {
        for (int a = 0; a < actions; ++a)
        {
            if (!first)
                cout << ", ";
            else
                first = false;
            vector<double> xs = cmp.features(s, a);
            double y = vi.Q[s][a];
            cout << "{";
            for (auto x : xs)
            {
                cout << x << ", ";
            }
            cout << y << "}";
        }
    }
    cout << "}" << endl;
    */

    vector<double> phi = cmp.features(2, 0);
    cout << "phi(s=2, a=0) = ";
    for (auto f : phi)
    {
        cout << f << " ";
    }
    cout << endl;
    auto tp = cmp.kernel->getTransitionProbabilities(2, 0);
    for (auto p : tp)
    {
        cout << "P(s'=" << p.first << " | s=2, a=0) = " << p.second << "\t";
    }
    cout << endl;



    ConstPolicy pi0(0, states);
    vector<double> w0 = test_lstdq_randomMDP(mdp, pi0);

    set<int> demoStates;
    for (Demonstration demo : demonstrations)
        for (Transition tr : demo)
            demoStates.insert(tr.s);

    double loss = BMT::loss(w0, lspiPolicy.getWeights(), demoStates, cmp);
    cout << "Loss between pi(.)=a0 and pi_lspi(.) compared over "
         << demoStates.size() << " states is: " << loss << endl;
}

/*

void test_discretecmp()
{
    cout << "*** Testing DiscreteCMP..." << endl;
    DiscreteCMP cmp(10, 10); //
    cmp.setTransitionProbability(0, 0, 0, 0.7);
    double prob = cmp.getTransitionProbability(0, 0, 0);
    std::cout << "Prob: " << prob << std::endl;
}


DiscreteMDP getGridMDP(int cmp_size)
{
    GridCMP *cmp = new GridCMP(cmp_size); // 10x10 grid, 100 states.
    DiscreteMDP mdp(cmp, 0.99);

    for (int s = 0; s < cmp->states; ++s)
        mdp.setReward(s, -1);
    mdp.setReward(cmp_size - 1, 100); // top right state has reward 100, rest -1
    // Make top right state terminal
    for (int s2 = 0; s2 < cmp->states; ++s2)
        for (int a = 0; a < cmp->actions; ++a)
            cmp->setTransitionProbability(cmp_size - 1, a, s2, 0);

    return mdp;
}

void printGridCMP_Q(ValueIteration &vi, int cmp_size, int actions)
{
    cout.precision(numeric_limits< double >::digits10 - 12);
    for (int y = 0; y < cmp_size; ++y)
    {
        for (int x = 0; x < cmp_size; ++x)
        {
            int s = x + y * cmp_size;
            double Q_max = -DBL_MAX;
            for (int a = 0; a < actions; ++a)
                if (vi.Q[s][a] > Q_max)
                    Q_max = vi.Q[s][a];
            cout << fixed << Q_max << " | \t";
        }
        cout << endl;
    }
    cout << endl;
}

void printGridCMP_V(ValueIteration &vi, int cmp_size)
{
    cout.precision(numeric_limits< double >::digits10 - 12);
    for (int y = 0; y < cmp_size; ++y)
    {
        for (int x = 0; x < cmp_size; ++x)
        {
            int s = x + y * cmp_size;
            cout << fixed << vi.V[s] << " | \t";
        }
        cout << endl;
    }
    cout << endl;
}

void compare_vi_qi(int cmp_size, double epsilon)
{
    cout << "*** Comparing timing of ValueIteration (V/Q) on GridCMP(" << cmp_size << ")" << endl;

    DiscreteMDP mdp = getGridMDP(cmp_size);

    ValueIteration vi(&mdp);

    auto t1_v = std::chrono::high_resolution_clock::now();
    // vi.computeStateValues(epsilon);
    auto t2_v = chrono::high_resolution_clock::now();
    auto time_v =
        chrono::duration_cast<chrono::milliseconds>(t2_v-t1_v).count();

    auto t1_q = std::chrono::high_resolution_clock::now();
    // vi.computeStateActionValues(epsilon);
    auto t2_q = chrono::high_resolution_clock::now();
    auto time_q =
        chrono::duration_cast<chrono::milliseconds>(t2_q-t1_q).count();

    cout << "t(V) = " << time_v << " ms" << endl;
    cout << "t(Q) = " << time_q << " ms" << endl;

    delete(mdp.cmp);
}
*/
