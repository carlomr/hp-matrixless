
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/geometry_info.h>
#include "gaussjacobi.h"

#include <cmath>
#include <limits>
#include <algorithm>
DEAL_II_NAMESPACE_OPEN

namespace {
    long double gammln(const long double xx) 
    { 
	long double x,tmp,y,ser; 
	static const long double cof[14]={57.1562356658629235,-59.5979603554754912, 
	14.1360979747417471,-0.491913816097620199,.339946499848118887e-4, 
	.465236289270485756e-4,-.983744753048795646e-4,.158088703224912494e-3, 
	-.210264441724104883e-3,.217439618115212643e-3,-.164318106536763890e-3, 
	.844182239838527433e-4,-.261908384015814087e-4,.368991826595316234e-5}; 
	if (xx <= 0) throw("bad arg in gammln"); 
	y=x=xx; 
	tmp = x+5.24218750000000000; 
	tmp = (x+0.5)*log(tmp)-tmp; 
	ser = 0.999999999999997092; 
	for (unsigned int j=0;j<14;j++) ser += cof[j]/++y; 
	return tmp+log(2.5066282746310005*ser/x); 
    } 
    template <typename number>
    number abs (const number a)
    {
	return ((a>0) ? a : -a);
    }

}

template <>
QGaussJacobi<0>::QGaussJacobi (const unsigned int n,
				const long double alpha,
				const long double beta)
  :
  // there are n_q^dim == 1
  // points
  Quadrature<0> (1)
{
  // the single quadrature point gets unit
  // weight
  this->weights[0] = 1;
}





template <>
QGaussJacobi<1>::QGaussJacobi (const unsigned int n, 
				const long double alf,
				const long double bet)
:
Quadrature<1> (n)
{
  if (n == 0)
    return;

  const long double
      long_double_eps = static_cast<long double>(std::numeric_limits<long double>::epsilon()),
	  double_eps      = static_cast<long double>(std::numeric_limits<double>::epsilon());

  volatile long double runtime_one = 1.0;
  const long double tolerance
      = (runtime_one + long_double_eps != runtime_one
	      ?
	      std::max (double_eps / 100, long_double_eps * 5)
	      :
	      double_eps * 5
	);


  long double alfbet,an,bn,r1,r2,r3;
  long double a,b,c,p,p1,p2,p3,pp,temp,z;
  std::vector<long double> x(n);
//TODO: prendere un'altra versione, questa è scandalosa (e protetta)
  for (unsigned int i=1;i<=n;++i)
  {
      if (i == 1) {
	  an=alf/n;
	  bn=bet/n;
	  r1=(1.0+alf)*(2.78/(4.0+n*n)+0.768*an/n);
	  r2=1.0+1.48*an+0.96*bn+0.452*an*an+0.83*an*bn;
	  z=1.0-r1/r2;
      } else if (i == 2) {
	  r1=(4.1+alf)/((1.0+alf)*(1.0+0.156*alf));
	  r2=1.0+0.06*(n-8.0)*(1.0+0.12*alf)/n;
	  r3=1.0+0.012*bet*(1.0+0.25*abs(alf))/n;
	  z -= (1.0-z)*r1*r2*r3;
      } else if (i == 3) {
	  r1=(1.67+0.28*alf)/(1.0+0.37*alf);
	  r2=1.0+0.22*(n-8.0)/n;
	  r3=1.0+8.0*bet/((6.28+bet)*n*n);
	  z -= (x[1]-z)*r1*r2*r3;
      } else if (i == n-1) {
	  r1=(1.0+0.235*bet)/(0.766+0.119*bet);
	  r2=1.0/(1.0+0.639*(n-4.0)/(1.0+0.71*(n-4.0)));
	  r3=1.0/(1.0+20.0*alf/((7.5+alf)*n*n));
	  z += (z-x[n-3])*r1*r2*r3;
      } else if (i == n) {
	  r1=(1.0+0.37*bet)/(1.67+0.28*bet);
	  r2=1.0/(1.0+0.22*(n-8.0)/n);
	  r3=1.0/(1.0+8.0*alf/((6.28+alf)*n*n));
	  z += (z-x[n-2])*r1*r2*r3;
      } else {
	  z=3.0*x[i-1]-3.0*x[i-2]+x[i-3];
      }
      alfbet=alf+bet;
      do
      {	
	  temp=2.0+alfbet;
	  p1=(alf-bet+temp*z)/2.0;
	  p2=1.0;
	  for (unsigned int j=2;j<=n;++j) 
	  {
	      p3=p2;
	      p2=p1;
	      temp=2*j+alfbet;
	      a=2*j*(j+alfbet)*(temp-2.0);
	      b=(temp-1.0)*(alf*alf-bet*bet+temp*(temp-2.0)*z);
	      c=2.0*(j-1+alf)*(j-1+bet)*temp;
	      p1=(b*p2-c*p3)/a;
	  }
	  pp=(n*(alf-bet-temp*z)*p1+2.0*(n+alf)*(n+bet)*p2)/(temp*(1.0-z*z));
	  z=z-p1/pp;
      } while (abs(p1/pp) > tolerance);
      x[i] = z;
      this->quadrature_points[i] = Point<1>(static_cast<double>(z));
      this->weights[i]=exp(gammln(alf+n)+gammln(bet+n)-gammln(n+1.0)-
	      gammln(n+alfbet+1.0))*temp*pow(2.0,alfbet)/(pp*p2);
  }
}

template <int dim>
QGaussJacobi<dim>::QGaussJacobi (const unsigned int n, const long double alf, const long double bet
  :  Quadrature<dim> (QGaussJacobi<dim-1>(n,alf, bet), QGaussJacobi<1>(n, alf, bet))
{}


template class QGaussJacobi<2>;
template class QGaussJacobi<3>;
DEAL_II_NAMESPACE_CLOSE
