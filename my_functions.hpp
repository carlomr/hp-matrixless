#ifndef __MY_FUNCTIONS_HPP__
#define __MY_FUNCTIONS_HPP__

#include <deal.II/base/function_lib.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <vector>
#include <deal.II/lac/vector.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor.h>

namespace MyFun
{

  using dealii::Point;
  using dealii::Tensor;
  using dealii::StandardExceptions::ExcDimensionMismatch;

  template<int dim>
  class MyExactSolution : public dealii::Function<dim>
  {
    private:
      double alpha;
      Point<dim> singularity;
    public:
      double get_alpha()const{return alpha;}
      Point<dim> get_singularity()const{return singularity;}
      MyExactSolution(const double a, const Point<dim> &s = Point<dim>(), const unsigned int n_components=1);

      double value(const Point<dim>& p, const unsigned int component=0 )const;
      void value_list(const std::vector<Point<dim> > &points, 
	  std::vector<double> &values, 
	  const unsigned int component = 0)const;

      double laplacian (const Point<dim> &p,
	  const unsigned int  component = 0) const;
      void laplacian_list (const std::vector<Point<dim> > &points,
       	std::vector<double> &values,
       	const unsigned int component = 0) const;

      Tensor<1,dim> gradient (const Point<dim>   &p,
	  const unsigned int  component = 0) const;
      void gradient_list (const std::vector<Point<dim> > &points,
                                std::vector<Tensor<1,dim> >    &gradients,
                                const unsigned int              component = 0) const;

  };
  template <int dim>
    MyExactSolution<dim>::MyExactSolution (const double a, const Point<dim> &s, const unsigned int n_components)
    :
      dealii::Function<dim> (n_components),
      alpha(a),
      singularity(s)
  {}



  template<int dim>
    double
    MyExactSolution<dim>::value (const Point<dim>   &p,
	const unsigned int) const
    {
      return pow(p.distance(singularity),alpha);
      //return std::cos(M_PI* p[0])*std::cos(M_PI*p[1]);
    }

  template<int dim>
    void
    MyExactSolution<dim>::value_list (const std::vector<dealii::Point<dim> > &points,
	std::vector<double> &values,
	const unsigned int) const
    {
      Assert (values.size() == points.size(),
	  ExcDimensionMismatch(values.size(), points.size()));

      for (unsigned int i=0; i<points.size(); ++i)
	values[i] = value(points[i]);
    }


  //XXX: ci devo mettere il meno o no?
  template<int dim>
    double
    MyExactSolution<dim>::laplacian (const Point<dim>   &p,
	const unsigned int) const
    {
      const double r = p.distance(singularity);
      //return -alpha*(alpha-2+dim)*pow(r,alpha-2);
      return -alpha*(alpha-2+dim)*pow(r,alpha-2)+pow(r,alpha);
      //return 2*M_PI*M_PI*std::cos(M_PI * p[0])*std::cos(M_PI * p[1]);
    }

  template<int dim>
    void
    MyExactSolution<dim>::laplacian_list (const std::vector<Point<dim> > &points,
	std::vector<double>            &values,
	const unsigned int) const
    {
      Assert (values.size() == points.size(),
	  ExcDimensionMismatch(values.size(), points.size()));

      for (unsigned int i=0; i<points.size(); ++i)
	values[i] = laplacian(points[i]);
    }

  template<int dim>
    Tensor<1,dim>
    MyExactSolution<dim>::gradient (const Point<dim>   &p,
	const unsigned int) const
    {
      Tensor<1,dim> result;
      for(unsigned int i=0; i<dim; ++i)
      {
	result[i] = alpha*pow(p.distance(singularity), alpha-2.)*p[i];
      }
      return result;
    }

  template<int dim>
    void
    MyExactSolution<dim>::gradient_list (const std::vector<Point<dim> > &points,
	std::vector<Tensor<1,dim> >    &gradients,
	const unsigned int) const
    {
      Assert (gradients.size() == points.size(),
	  ExcDimensionMismatch(gradients.size(), points.size()));

      for (unsigned int i=0; i<points.size(); ++i)
      {
	const Point<dim> &p = points[i];
	const double r = p.distance(singularity);
	for(unsigned int k=0; k<dim; ++k)
	  gradients[i][k] = alpha*pow(r, alpha-2)*p[k];
      }
    }
}

#endif
