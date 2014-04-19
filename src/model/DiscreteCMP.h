#ifndef DISCRETECMP_H
#define DISCRETECMP_H

#include <vector>

/*
 * Distinguish between MDP and CMP (Controlled Markov Process) to make multi
 * task (multi MDP) IRL more explicitly stated.
 */
class DiscreteCMP
{
    public:
        DiscreteCMP(int _states, int _actions);
        double getTransitionProbability(const int s,
                                        const int a,
                                        const int s2) const;
        void setTransitionProbability(const int s,
                                      const int a,
                                      const int s2,
                                      const double p);
        const int states;
        const int actions;
    private:
        std::vector<std::vector<std::vector<double>>> kernel;
};

#endif
