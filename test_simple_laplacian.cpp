#include<iostream>
#include<functional>
#include<cmath>
#include"bases.hpp"


int main()
{
    const int N=10;
    auto f = [](double const & x, double const & y){return cos(x)*sin(y);};
    ModalBasis<N> B;
    auto f_int = B.Integrate(f);
    auto f_lap = B.ReferenceLaplacian(f_int);

    //std::cout << (f_lap-f_int) << std::endl;
    return 0;
}
