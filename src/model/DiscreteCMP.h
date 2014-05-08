#ifndef DISCRETECMP_H
#define DISCRETECMP_H

#include <vector>
#include "TransitionKernel.h"

typedef int state;
typedef int action;
typedef double probability;

/*
 * Distinguish between MDP and CMP (Controlled Markov Process) to make multi
 * task (multi MDP) IRL more explicitly stated.
 */
class DiscreteCMP
{
    public:
        DiscreteCMP(TransitionKernel const *_kernel);
        DiscreteCMP();
        const int states;
        const int actions;
        TransitionKernel const * const kernel;
        virtual ~DiscreteCMP();
        virtual std::vector<double> features(int s, int a) const;
        virtual std::vector<double> features(int s) const;
        virtual int nFeatures() const;
        virtual bool isTerminal(int s) const { return false; }
    private:
};

#endif
