# H2O PES
This program provides a C++ interface for water potential energy surface designed by H. Partridge and D. W. Schwenke.

* Author: Haoyu Lin
* E-mail: vileoy@pku.edu.cn
* Repo: [uovie/h2opes](https://github.com/uovie/h2opes)

## How to build:

Just `make` it.

## Interface

```c++
class h2opes {
public:
    const double& V() const { return v; }                // potential energy V
    const Eigen::ArrayXd& dV() const { return dv; }      // gradient of V
    const Eigen::ArrayXXd& ddV() const { return ddv; }   // second derivative of V
    void calc(const Eigen::ArrayXd& q, const int& flag); // calculate
};
```
