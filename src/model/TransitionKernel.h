#ifndef TRANSITIONKERNEL_H
#define TRANSITIONKERNEL_H

#include <vector>

typedef int state;
typedef int action;
typedef double probability;

class TransitionKernel
{
    public:
        TransitionKernel(int _states, int _actions);
        virtual ~TransitionKernel() {};
        virtual double getTransitionProbability(const int s,
                                                const int a,
                                                const int s2) const = 0;

        virtual std::vector<int> getValidActions(const int s) const;

        std::vector< std::pair<state, probability> >
            getTransitionProbabilities(const int s, const int a) const;

        const int states;
        const int actions;
    private:
};

#endif
