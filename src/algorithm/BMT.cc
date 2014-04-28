#include "BMT.h"

#include <vector>
#include <cfloat>
#include <cmath>
#include <numeric>

#include "../model/DiscreteCMP.h"

using std::vector;
using std::inner_product;

double BMT::loss(vector<double> const & wEval,
                 vector<double> const & wOpt,
                 vector<int> const & states, DiscreteCMP const & cmp)
{
    double sup = -DBL_MAX;
    // TODO: Do this for all state action pairs in a demonstration instead?
    for (int s : states)
    {
        vector<double> phi = cmp.features(s);
        double vOpt = inner_product(phi.begin(), phi.end(), wEval.begin(), 0.0);
        double vEval = inner_product(phi.begin(), phi.end(), wOpt.begin(), 0.0);
        double diff = fabs(vOpt - vEval);
        if (diff > sup)
            sup = diff;
    }

    return sup;
}
