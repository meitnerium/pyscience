#include <Eigen/Dense>

class h2opes {
public:
    h2opes() = default;
    
    const double& V() const { return v; }
    const Eigen::ArrayXd& dV() const { return dv; }
    const Eigen::ArrayXXd& ddV() const { return ddv; }
    void calc(const Eigen::ArrayXd& q, const int& flag);

private:
    double v;
    Eigen::ArrayXd dv;
    Eigen::ArrayXXd ddv;
};