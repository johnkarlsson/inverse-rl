#ifndef TABULARTRANSITIONKERNEL_H
#define TABULARTRANSITIONKERNEL_H

#include "TransitionKernel.h"
#include <vector>
#include <set>

class TabularTransitionKernel
    : public TransitionKernel
{
    public:
        TabularTransitionKernel(int _states, int _actions);

        ~TabularTransitionKernel();

        double getTransitionProbability(const int s,
                                        const int a,
                                        const int s2) const;

        std::set<int> getValidActions(const int s) const;

        std::vector< std::pair<state, probability> >
            getTransitionProbabilities(const int s, const int a) const;

        void setTransitionProbability(const int s,
                                      const int a,
                                      const int s2,
                                      const double p);
    private:
        std::vector<std::vector<std::vector<double>>> kernel;
        std::vector<std::set<int>> valid_actions;
};

#endif
