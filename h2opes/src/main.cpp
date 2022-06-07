#include <iostream>
#include "pes.h"

int main()
{
    h2opes h2o;
    Eigen::ArrayXd q(9);
    q << 1.48123672, 1.04739254, 0.00000000,
         0.00000000, 0.00000000, 0.00000000,
        -1.48123672, 1.04739254, 0.00000000;

    h2o.calc(q, 0);
    std::cout << h2o.V() << std::endl;

    return 0;
}