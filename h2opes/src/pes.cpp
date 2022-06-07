#include "pes.h"

// Fortran Subroutine: PES for H2O
extern "C" {
    void h2opot(double* v, double dv[9], double ddv[9][9], double cart[9], const int* flag);
}

void h2opes::calc(const Eigen::ArrayXd& q, const int& flag)
{
    double _dv[9];
    double _ddv[9][9];
    double _cart[9];

    for (int i = 0; i < 9; i++)
        _cart[i] = q(i);

    h2opot(&v, _dv, _ddv, _cart, &flag);

    if (flag > 0) {
        dv.resize(9);
        for (int i = 0; i < 9; i++)
            dv(i) = _dv[i];
        if (flag > 1) {
            ddv.resize(9, 9);
            for (int i = 0; i < 9; i++)
                for (int j = 0; j < 9; j++)
                    ddv(i, j) = _ddv[i][j];
        }
    }
}