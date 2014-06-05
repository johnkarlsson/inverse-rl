#ifndef TRANSITION_H
#define TRANSITION_H

#include <vector>

class Transition
{
    public:
        Transition() {};
        Transition(int _s, int _a, int _s2, double _r);
        // Transition(int _s, int _a);
        int s, a;
        double s2, r; // Not used in LSTDQ "with a model",
                      // but useful for MC playouts.
    private:
};

typedef std::vector<Transition> Demonstration;

#endif
