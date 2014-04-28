#ifndef BMT_H
#define BMT_H

#include <vector>

#include "../model/DiscreteCMP.h"

using std::vector;

class BMT
{
    public:
        static double loss(vector<double> const & weightsEval,
                           vector<double> const & weightsOpt,
                           vector<int> const & states, DiscreteCMP const & cmp);
    private:
};

#endif
