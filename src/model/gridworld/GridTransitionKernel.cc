#include "GridTransitionKernel.h"

int stateIndex(int x, int y, int size)
{
    return x+y*size;
}

GridTransitionKernel::GridTransitionKernel(int size)
    : TabularTransitionKernel(size*size, 4)
{
    // int i = 0;
    // Initialize grid transition probabilities.
    for (int x = 0; x < size; ++x)
        for (int y = 0; y < size; ++y)
        {
            int s = stateIndex(x, y, size);
            int xn[] = { x,   x+1, x,   x-1 };
            int yn[] = { y+1, y,   y-1, y   };
            for (int ai = 0; ai < 4; ++ai) // up, right, down, left
            {
                    int s2 = stateIndex(xn[ai], yn[ai], size);
                    if (xn[ai] >= 0 && xn[ai] < size && 
                        yn[ai] >= 0 && yn[ai] < size)
                    {
                        setTransitionProbability(s, ai, s2, 1);
                    }
            }
        }
}
