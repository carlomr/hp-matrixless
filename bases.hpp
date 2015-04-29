#ifndef __BASES_HPP__
#define __BASES_HPP__

#include<vector>
#include<Eigen/SparseCore>
#include<Eigen/Core>
#include<Eigen/Dense>
#include<array>
#include<algorithm>
namespace{

    long double JacobiP(const long double x,
	    const int alpha,
	    const int beta,
	    const unsigned int n)
    {
	// the Jacobi polynomial is evaluated
	// using a recursion formula.
	std::vector<long double> p(n+1);
	int v, a1, a2, a3, a4;

	// initial values P_0(x), P_1(x):
	p[0] = 1.0L;
	if (n==0) return p[0];
	p[1] = ((alpha+beta+2)*x + (alpha-beta))/2;
	if (n==1) return p[1];

	for (unsigned int i=1; i<=(n-1); ++i)
	{
	    v  = 2*i + alpha + beta;
	    a1 = 2*(i+1)*(i + alpha + beta + 1)*v;
	    a2 = (v + 1)*(alpha*alpha - beta*beta);
	    a3 = v*(v + 1)*(v + 2);
	    a4 = 2*(i+alpha)*(i+beta)*(v + 2);

	    p[i+1] = static_cast<long double>( (a2 + a3*x)*p[i] - a4*p[i-1])/a1;
	} // for
	return p[n];
    }
    template<int N>
	void gauleg(Eigen::Array<double,N,1> &w, Eigen::Array<double,N,1> &x, double a, double b)
	{
	    int n=N;
	    double m=(n+1.)/2.0f;
	    double xm=0.5*(b+a);
	    double xl=0.5*(b-a);

	    double z, z1, pi=M_PI, p1, p2, p3, pp;
	    double eps = std::numeric_limits<double>::epsilon();
	    //double eps = 1e-28;

	    for(int i=1; i<=m; ++i){
		z=cos(pi*(i-0.25)/(n+0.5));
		do{
		    p1=1.0;
		    p2=0.0;
		    for(int j=1; j<=n; ++j)
		    {
			p3=p2;
			p2=p1;
			p1=((2.0*j-1.0)*z*p2-(j-1.0)*p3)/j;
		    }
		    pp=n*(z*p1-p2)/(z*z-1.0);
		    z1=z;
		    z=z1-p1/pp;
		}while(std::abs(z-z1)>eps);
		x(i-1)=xm-xl*z;
		x(n-i)=xm+xl*z;
		//std::cout << n-i << ": " << x(n-i) << std::endl;
		w(i-1)=2.0*xl/((1.0-z*z)*pp*pp);
		w(n-i)=w(i-1);
		//std::cout << "=============================="<<std::endl;
		//std::cout << i << std::endl;
		//std::cout << w << std::endl;
	    }
	    //x=xx;
	    //w=ww;
	}

    Eigen::ArrayXd jacobi_polynomial(int N, int alpha, int beta, Eigen::ArrayXd const & z)
    {
	typedef Eigen::ArrayXd  Arr;
	if(N==0)
	    return Arr::Ones(z.size());
	if(N==1)
	    return 0.5*(alpha - beta + (alpha + beta + 2.0)*z);

	unsigned int dims = z.size();
	//Arr jf = Arr::Zero(dims);
	const float one = 1.0;
	const float two = 2.0;

	int apb = alpha + beta;

	Arr poly   = Arr::Zero(dims);
	Arr polyn2 = Arr::Ones(dims);
	Arr polyn1 = 0.5*(alpha - beta + (alpha + beta + two)*z);

	double a1, a2, a3, a4;
	for(int k=2; k<=N;++k)
	{
	    a1 =  two*k*(k + apb)*(two*k + apb - two);
	    a2 = (two*k + apb - one)*(alpha*alpha - beta*beta);
	    a3 = (two*k + apb - two)*(two*k + apb - one)*(two*k + apb);
	    a4 =  two*(k + alpha - one)*(k + beta - one)*(two*k + apb);

	    a2 = a2/a1;
	    a3 = a3/a1;
	    a4 = a4/a1;

	    poly   = (a2 + a3*z)*polyn1 - a4*polyn2;
	    polyn2 = polyn1;
	    polyn1 = poly;
	}
	return poly;
    }

}


/*template<int N>*/
//class MonoDimensionalMatrix
//{
    //public:
	//MonoDimensionalMatrix(){M_bandwidth=N;};
	//virtual ~MonoDimensionalMatrix()=0;
    //protected:
	//unsigned int M_bandwidth;
//};
template<int N>
class ModalBasis
{
    public:
	ModalBasis();
	typedef Eigen::Matrix<double,N+1, N+1> Mat;
	typedef Eigen::Matrix<double,(N+1)*(N+1),1> Vec;
	Mat OneDimMass;
	Mat OneDimLap;
	Vec ReferenceLaplacian(const Vec & u)const;
	Vec Integrate(std::function<double(const double &, const double &)> const & f)const;
	static const int half_bandwidth = 3;
    private:
	//typedef std::array<double,N> Arr;
	static constexpr int num_q = N+1;
	typedef Eigen::Array<double, num_q,1> Arr;
	//typedef Eigen::ArrayXd Arr;
	std::vector<std::function<Arr(const Arr&)>> M_phi;
	std::vector<std::function<Arr(const Arr&)>> M_grad;
	Eigen::Matrix<double,N+1,num_q> M_phi_x_w;
	Arr M_x_quad;

};

template<int N>
ModalBasis<N>::ModalBasis():OneDimMass(N+1,N+1), OneDimLap(N+1,N+1)
{
    //const int num_q = N+1;

    std::vector<Eigen::Triplet<double>> MtripletList;
    std::vector<Eigen::Triplet<double>> GtripletList;
    MtripletList.reserve(5*N);
    GtripletList.reserve(N+2);

    M_phi.push_back([](Arr const & x)->Arr{return (1+x)/2;});
    M_grad.push_back([](Arr const & x)->Arr{return 1./2*Arr::Ones(x.size());});
    M_phi.push_back([](Arr const & x)->Arr{return (1-x)/2;});
    M_grad.push_back([](Arr const & x)->Arr{return -1./2*Arr::Ones(x.size());});
    for(int i=2; i<=N; ++i)
    {
	M_phi.push_back([i](Arr const & x)->Arr{return (1+x)*(1-x)/4*jacobi_polynomial(i-2,1,1,x);});
	if(i==2)
	    M_grad.push_back([i](Arr const & x)->Arr{return -2*x/4*jacobi_polynomial(i-2,1,1,x);});
	else
	    M_grad.push_back([i](Arr const & x)->Arr{return (1+x)*(1-x)/4*jacobi_polynomial(i-3,2,2,x)*(i+1)/2
		    -2*x/4*jacobi_polynomial(i-2,1,1,x);});
    }

    Eigen::Array<double,num_q,1> w;
    Eigen::Array<double,num_q,1> x;
    gauleg<num_q>(w,x, -1, 1);

    M_x_quad = x;
    OneDimMass = Mat::Zero(N+1,N+1);
    OneDimLap =  Mat::Zero(N+1,N+1);

    for (unsigned int i=0; i<=N; ++i)
    {
	M_phi_x_w.row(i) = M_phi[i](x)*w;
	for (unsigned int j=0; j<=N; ++j)
	{
	    double Msum = (w*M_phi[i](x) *M_phi[j](x)).sum();
	    double Gsum = (w*M_grad[i](x)*M_grad[j](x)).sum();
	    if(std::fabs(Msum)>1e-6)
		OneDimMass(i,j) = Msum;
		//MtripletList.emplace_back(i,j,Msum);
	    if(std::fabs(Gsum)>1e-6)
		OneDimLap(i,j) = Gsum;
		//GtripletList.emplace_back(i,j,Gsum);
	}
    }
    //OneDimMass.setFromTriplets(MtripletList.begin(), MtripletList.end());
    //OneDimLap.setFromTriplets(GtripletList.begin(), GtripletList.end());
}

template<int N>
typename ModalBasis<N>::Vec ModalBasis<N>::ReferenceLaplacian(const Vec & u)const
{
    Eigen::Matrix<double, N+1, N+1> du;
    Eigen::Matrix<double, N+1, N+1> ud;
    Vec out;
    for(size_t k = 0 ; k<N+1; ++k)
	for(size_t j = 0 ; j<N+1; ++j)
	{
	    unsigned int idx = (N+1)*k+j;
	    du(k,j) = OneDimLap(k,k)*u(idx);
	    ud(k,j) = OneDimLap(j,j)*u(idx);
	}
    for(size_t j=0; j<N+1; ++j)
    {
	du(0,j) += OneDimLap(0,1) * u(1*(N+1)+j);
	du(1,j) += OneDimLap(1,0) * u(0*(N+1)+j);
	ud(j,0) += OneDimLap(1,0) * u(j*(N+1)+1);
	ud(j,1) += OneDimLap(0,1) * u(j*(N+1)+0);
    }
    for(unsigned int k = 0 ; k<N+1; ++k)
    {
	for (int l = 0; l<N+1; ++l)
	{
	    unsigned int idx = (N+1)*k+l;
	    for(unsigned int j=std::max(0,l-half_bandwidth); j< std::min(N+1,l+half_bandwidth); ++j)
	    {
		out(idx) += du(k,j)*OneDimMass(j,l) + OneDimMass(k,j)+ud(j,l);
	    }
	}
    }
    return out;
}
template<int N>
typename ModalBasis<N>::Vec 
ModalBasis<N>::Integrate(std::function<double(const double &, const double &)> const & f)const
{
    Vec out = Vec::Zero((N+1)*(N+1));
    Mat a =  Mat::Zero(N+1,N+1);
    for(unsigned int i=0; i<N+1; ++i)
	for(unsigned int n = 0; n<N+1; ++n)
	    for(unsigned int k = 0; k<num_q; ++k)
		a(i,n) += f(M_x_quad(k), M_x_quad(n))*M_phi_x_w(k);
    for(unsigned int i=0; i<N+1; ++i)
	    for(unsigned int k = 0; k<num_q; ++k)
		out(i) += M_phi_x_w(k)*a(i,k);
    return out;
}
#endif
