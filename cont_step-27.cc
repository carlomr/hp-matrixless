/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2006 - 2013 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, Texas A&M University, 2006, 2007
 */


// @sect3{Include files}

// The first few files have already been covered in previous examples and will
// thus not be further commented on.
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/matrix_out.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_cartesian.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/integrators/laplace.h>

// These are the new files we need. The first one provides an alternative to
// the usual SparsityPattern class and the CompressedSparsityPattern class
// already discussed in step-11 and step-18. The last two provide <i>hp</i>
// versions of the DoFHandler and FEValues classes as described in the
// introduction of this program.
#include <deal.II/lac/compressed_set_sparsity_pattern.h>
#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_values.h>

// The last set of include files are standard C++ headers. We need support for
// complex numbers when we compute the Fourier transform.
#include <fstream>
#include <iostream>
#include <complex>

#include "my_functions.hpp"
#include <deal.II/base/gaussjacobi.h>

#define MAX(a,b) (a>b)?a:b

#define DEBUG_P(a) std::cout << #a <<" = " << a << std::endl;

// Finally, this is as in previous programs:
namespace Step27
{
  using namespace dealii;


  // @sect3{The main class}

  // The main class of this program looks very much like the one already used
  // in the first few tutorial programs, for example the one in step-6. The
  // main difference is that we have merged the refine_grid and output_results
  // functions into one since we will also want to output some of the
  // quantities used in deciding how to refine the mesh (in particular the
  // estimated smoothness of the solution). There is also a function that
  // computes this estimated smoothness, as discussed in the introduction.
  //
  // As far as member variables are concerned, we use the same structure as
  // already used in step-6, but instead of a regular DoFHandler we use an
  // object of type hp::DoFHandler, and we need collections instead of
  // individual finite element, quadrature, and face quadrature objects. We
  // will fill these collections in the constructor of the class. The last
  // variable, <code>max_degree</code>, indicates the maximal polynomial
  // degree of shape functions used.
  template <int dim>
  class LaplaceProblem
  {
  public:
    LaplaceProblem ();
    ~LaplaceProblem ();

    void run ();

  private:
    void setup_system ();
    void assemble_system ();
    void solve ();
    void create_coarse_grid ();
    void estimate_smoothness (Vector<float> &smoothness_indicators) const;
    void postprocess (const unsigned int cycle);
    void error();

    void boundary_term(FullMatrix<double> &lhs_mat, Vector<double> &rhs_vec,
			 const FEValuesBase<dim> &fe, const std::vector<double> &bound_values, 
			 const double &penalty);

    void face_term(typename hp::DoFHandler<dim>::cell_iterator cell,
			    typename hp::DoFHandler<dim>::cell_iterator neighbor,
			    const FEValuesBase<dim> & fe_cell,
			    const FEValuesBase<dim> & fe_neighbor,
			    FullMatrix<double> & cell_matrix,
			    FullMatrix<double> & cell_neigh_matrix,
			    FullMatrix<double> & neigh_cell_matrix,
			    FullMatrix<double> & neigh_matrix,
			    const double & penalty);

    void cell_rhs(Vector<double> & rhs_vec, const FEValuesBase<dim> & fe, 
		    const std::vector<double> & rhs_values);

    Triangulation<dim>   triangulation;
    MyFun::MyExactSolution<dim> exact_solution;

    hp::DoFHandler<dim>        dof_handler;
    hp::FECollection<dim>      fe_collection;
    hp::QCollection<dim>       quadrature_collection;
    hp::QCollection<dim-1>     face_quadrature_collection;
    hp::MappingCollection<dim> mapping_collection;

    ConstraintMatrix     constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double>       solution;
    Vector<double>       system_rhs;

    const unsigned int max_degree;
    double L2_error;
  };




  // @sect3{Implementation of the main class}

  // @sect4{LaplaceProblem::LaplaceProblem}

  // The constructor of this class is fairly straightforward. It associates
  // the hp::DoFHandler object with the triangulation, and then sets the
  // maximal polynomial degree to 7 (in 1d and 2d) or 5 (in 3d and higher). We
  // do so because using higher order polynomial degrees becomes prohibitively
  // expensive, especially in higher space dimensions.
  //
  // Following this, we fill the collections of finite element, and cell and
  // face quadrature objects. We start with quadratic elements, and each
  // quadrature formula is chosen so that it is appropriate for the matching
  // finite element in the hp::FECollection object.
  template <int dim>
  LaplaceProblem<dim>::LaplaceProblem ()
    :
    exact_solution(3./4.),
    dof_handler (triangulation),
    max_degree (12),//TODO:spostarne la definizione fuori
    L2_error(0)
  {
      unsigned int idx = 0;

      //QGaussJacobi<dim> prova1(3, 0, 0);
      //QGauss<dim> prova2(3);


      //for (unsigned int i=0; i<prova2.get_weights().size(); ++i)
      //{
	  //std::cout << prova1.weight(i) << std::endl;
	  //std::cout << prova2.weight(i) << std::endl;
	  //std::cout << "==============================" << std::endl;
	  //std::cout << prova1.point(i) << std::endl;
	  //std::cout << prova2.point(i) << std::endl;
	  //std::cout << "______________________________" << std::endl;
      //}
    for (unsigned int degree=2; degree<=max_degree; ++degree, ++idx)
      {
        fe_collection.push_back (FE_Q<dim>(degree));
	//if (idx == 0)
	    //quadrature_collection.push_back(QGaussJacobi<dim>(degree+1, 0, 0 )); //TODO: in ognuno dei quattro quadrati è diverso: come li scelgo? E' un bel casino
	    //quadrature_collection.push_back (QGaussOneOverR<dim>(degree+1, exact_solution.singularity, true));
	//else
	    quadrature_collection.push_back (QGauss<dim>(degree+1));//TODO: negli elementi centrali deve essere Jacobi

        face_quadrature_collection.push_back (QGauss<dim-1>(max_degree+1));
	mapping_collection.push_back(MappingCartesian<dim>());
      }
  }


  // @sect4{LaplaceProblem::~LaplaceProblem}

  // The destructor is unchanged from what we already did in step-6:
  template <int dim>
  LaplaceProblem<dim>::~LaplaceProblem ()
  {
    dof_handler.clear ();
  }


  // @sect4{LaplaceProblem::setup_system}
  //
  // This function is again a verbatim copy of what we already did in
  // step-6. Despite function calls with exactly the same names and arguments,
  // the algorithms used internally are different in some aspect since the
  // dof_handler variable here is an hp object.
  template <int dim>
  void LaplaceProblem<dim>::setup_system ()
  {
    dof_handler.distribute_dofs (fe_collection);

    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());

    constraints.clear ();
    DoFTools::make_hanging_node_constraints (dof_handler,
					     constraints);
    VectorTools::interpolate_boundary_values (dof_handler,
					      0,
					      exact_solution,
					      constraints);
    constraints.close ();

    CompressedSetSparsityPattern csp (dof_handler.n_dofs(),
                                      dof_handler.n_dofs());
    DoFTools::make_flux_sparsity_pattern (dof_handler, csp, constraints, false);
    sparsity_pattern.copy_from (csp);

    system_matrix.reinit (sparsity_pattern);
  }


  template<int dim>
  void LaplaceProblem<dim>::boundary_term(FullMatrix<double> &lhs_mat, 
					      Vector<double> &rhs_vec,
					      const FEValuesBase<dim> &fe,
					      const std::vector<double> &boundary_values,
					      const double & penalty)
  {
      //XXX: quel 0.5 * penalty deve essercy perche' nitsche_matrix ha un inspiegabile 2
      LocalIntegrators::Laplace::nitsche_matrix( lhs_mat, fe, .5*penalty);


      for (unsigned k=0; k<fe.n_quadrature_points; ++k)
	  for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
	      rhs_vec(i) += 
		  (+ fe.shape_value(i,k) * penalty * boundary_values[k]
		   - fe.normal_vector(k) * fe.shape_grad(i,k) * boundary_values[k]) 
		  * fe.JxW(k);

  }


  template<int dim>
  void LaplaceProblem<dim>::cell_rhs(Vector<double> & rhs_vec, 
				      const FEValuesBase<dim> & fe, 
				      const std::vector<double> & rhs_values)
  {
      for (unsigned int q_point=0; q_point<fe.n_quadrature_points; ++q_point)
	  for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
	  {
              rhs_vec(i) += (fe.shape_value(i,q_point) *
                              rhs_values[q_point] *
                              fe.JxW(q_point));
	  }
  }

  template<int dim>
  void LaplaceProblem<dim>::face_term(typename hp::DoFHandler<dim>::cell_iterator cell, //XXX: vorrei fossero const_iterators, manon ci sono
			    typename hp::DoFHandler<dim>::cell_iterator neighbor,
			    const FEValuesBase<dim> & fe_cell, //NB: FE(Sub)FaceValues
			    const FEValuesBase<dim> & fe_neighbor,//NB: FE(Sub)FaceValues
			    FullMatrix<double> & cell_matrix,
			    FullMatrix<double> & cell_neigh_matrix,
			    FullMatrix<double> & neigh_cell_matrix,
			    FullMatrix<double> & neigh_matrix,
			    const double & penalty)
  {
      const unsigned int   dofs_per_cell = cell->get_fe().dofs_per_cell;
      const unsigned int   neigh_dofs = neighbor->get_fe().dofs_per_cell;

      cell_neigh_matrix.reinit (dofs_per_cell, neigh_dofs);
      neigh_cell_matrix.reinit ( neigh_dofs, dofs_per_cell);
      neigh_matrix.reinit ( neigh_dofs,neigh_dofs);

      cell_neigh_matrix = 0;
      neigh_cell_matrix = 0;
      neigh_matrix      = 0;

      LocalIntegrators::Laplace::ip_matrix(cell_matrix,
	      cell_neigh_matrix,
	      neigh_cell_matrix,
	      neigh_matrix,
	      fe_cell,
	      fe_neighbor,
	      penalty);
  }


  // @sect4{LaplaceProblem::assemble_system}
  template <int dim>
  void LaplaceProblem<dim>::assemble_system ()
  {
      //XXX: questi sarebbe simpatico capirli, magari non bisogna fare tutto?
    const UpdateFlags update_flags = update_values
                                     | update_gradients
                                     | update_quadrature_points
                                     | update_JxW_values;

    const UpdateFlags face_update_flags = update_values
					  | update_gradients
                                          | update_quadrature_points
                                          | update_JxW_values
                                          | update_normal_vectors;

    //const UpdateFlags neighbor_face_update_flags = update_values;

    hp::FEValues<dim>        hp_fe_values               (mapping_collection , fe_collection , quadrature_collection      , update_flags);
    hp::FEFaceValues<dim>    hp_fe_values_face          (mapping_collection , fe_collection , face_quadrature_collection , face_update_flags);
    hp::FESubfaceValues<dim> hp_fe_values_subface       (mapping_collection , fe_collection , face_quadrature_collection , face_update_flags);
    hp::FEFaceValues<dim>    hp_fe_values_face_neighbor (mapping_collection , fe_collection , face_quadrature_collection , face_update_flags); //era neighbor_


    //const RightHandSide<dim> rhs_function;

    FullMatrix<double>   cell_matrix;
    FullMatrix<double>   aux_cell_matrix;
    FullMatrix<double>   aux_cell_neigh_matrix;
    FullMatrix<double>   aux_neigh_cell_matrix;
    FullMatrix<double>   aux_neigh_matrix;
    Vector<double>       cell_rhs_vec;
    Vector<double>       aux_cell_rhs;

    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<types::global_dof_index> neighbor_dof_indices;

    for (auto cell :  dof_handler.active_cell_iterators())
      {
        //std::cout << cell->index() << " " << cell->center() << std::endl;
        const unsigned int   dofs_per_cell = cell->get_fe().dofs_per_cell;
	local_dof_indices.resize (dofs_per_cell);
	cell->get_dof_indices (local_dof_indices);

        cell_matrix.reinit (dofs_per_cell, dofs_per_cell);
        cell_matrix = 0;

        cell_rhs_vec.reinit (dofs_per_cell);
        cell_rhs_vec = 0;

        hp_fe_values.reinit (cell);
        const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values ();

        std::vector<double>  rhs_values (fe_values.n_quadrature_points);
        exact_solution.laplacian_list (fe_values.get_quadrature_points(),
                                 rhs_values);

	//cell element
	LocalIntegrators::Laplace::cell_matrix(cell_matrix, fe_values);
	//rhs cell element
	cell_rhs(cell_rhs_vec, fe_values, rhs_values);
      
	//std::cout << cell->center() << std::endl;
	/*for (unsigned int face_no=0; face_no<GeometryInfo<dim>::faces_per_cell; ++face_no)*/
	//{
	    //typename hp::DoFHandler<dim>::face_iterator face= cell->face(face_no);
	    //hp_fe_values_face.reinit (cell, face_no);
	    //const FEFaceValues<dim> &fe_values_face = hp_fe_values_face.get_present_fe_values (); 

	    ////aux_cell_rhs = 0;
	    //aux_cell_matrix = 0;
	    //// Case a)
	    //if (face->at_boundary())
	    //{
		//unsigned int deg = fe_values_face.get_fe().tensor_degree(); 
		//const double penalty = 2. * deg * (deg+1) * face->measure() / cell->measure();

		//std::vector<double> boundary_values(fe_values_face.n_quadrature_points);
		//exact_solution.value_list(fe_values_face.get_quadrature_points(), boundary_values);

		//boundary_term( cell_matrix,  cell_rhs_vec, fe_values_face, boundary_values,penalty);
	    //}
            //else
	    //{
		//Assert (cell->neighbor(face_no).state() == IteratorState::valid,
			//ExcInternalError());
                //typename hp::DoFHandler<dim>::cell_iterator neighbor=
                  //cell->neighbor(face_no);
                //// Case b), we decide that there are finer cells as neighbors
                //// by asking the face, whether it has children. if so, then
                //// there must also be finer cells which are children or
                //// farther offspring of our neighbor.
		////XXX:ha senso questo? dovendo fare solo una volta ogni termine, tanto vale che lo faccia nel caso più semplice -- se non che in step-30 lui fa solo questo
                //if (face->has_children())
		//{
		    //// We need to know, which of the neighbors faces points in
		    //// the direction of our cell. Using the @p
		    //// neighbor_face_no function we get this information for
		    //// both coarser and non-coarser neighbors.
		    //const unsigned int neighbor2=
		      //cell->neighbor_face_no(face_no);

		    //// Now we loop over all subfaces, i.e. the children and
		    //// possibly grandchildren of the current face.
		    //for (unsigned int subface_no=0;
			 //subface_no<face->number_of_children(); ++subface_no)
		    //{
			//// To get the cell behind the current subface we can
			//// use the @p neighbor_child_on_subface function. it
			//// takes care of all the complicated situations of
			//// anisotropic refinement and non-standard faces.
			//typename hp::DoFHandler<dim>::cell_iterator neighbor_child
			    //= cell->neighbor_child_on_subface (face_no, subface_no);
			//Assert (!neighbor_child->has_children(), ExcInternalError());

			//hp_fe_values_subface.reinit(cell, face_no, subface_no);
			//const FESubfaceValues<dim> &fe_values_subface = hp_fe_values_subface.get_present_fe_values (); //XXX:o dim-1?
			//hp_fe_values_face_neighbor.reinit(neighbor_child,neighbor2);
			//const FEFaceValues<dim> &fe_values_face_neighbor = hp_fe_values_face_neighbor.get_present_fe_values (); //XXX:o dim-1?



			//const unsigned int max_deg = MAX(fe_values_subface.get_fe().tensor_degree() ,
				//fe_values_face_neighbor.get_fe().tensor_degree());
			//const double min_h = neighbor_child->face(neighbor2)->diameter();
			//const double penalty = 2*max_deg*(max_deg+1)/min_h;

			//face_term(cell, neighbor_child, fe_values_subface,  fe_values_face_neighbor, cell_matrix,
				//aux_cell_neigh_matrix, aux_neigh_cell_matrix, aux_neigh_matrix, penalty);


			//const unsigned int   neigh_dofs = neighbor_child->get_fe().dofs_per_cell;
			//neighbor_dof_indices.resize (neigh_dofs);
			//neighbor_child->get_dof_indices (neighbor_dof_indices);

			//constraints.distribute_local_to_global(aux_cell_neigh_matrix,
				//local_dof_indices,
				//neighbor_dof_indices,
				//system_matrix);
			//constraints.distribute_local_to_global(aux_neigh_cell_matrix,
				//neighbor_dof_indices,
				//local_dof_indices,
				//system_matrix);
			//constraints.distribute_local_to_global(aux_neigh_matrix,
				//neighbor_dof_indices,
				//neighbor_dof_indices,
				//system_matrix);

		    //}
		//}
		//else
		//{
		    //// Case c). We simply ask, whether the neighbor is
		    //// coarser. If not, then it is neither coarser nor finer,
		    //if (!cell->neighbor_is_coarser(face_no) &&
			    //(neighbor->index() > cell->index() ||
			     //(neighbor->level() < cell->level() &&
			      //neighbor->index() == cell->index())))
		    //{
			//// Here we know, that the neighbor is not coarser so we
			//// can use the usual @p neighbor_of_neighbor
			//// function. However, we could also use the more
			//// general @p neighbor_face_no function.
			//const unsigned int neighbor2=cell->neighbor_of_neighbor(face_no);

			//hp_fe_values_face_neighbor.reinit(neighbor, neighbor2);
			//const FEFaceValues<dim> &fe_values_face_neighbor = hp_fe_values_face_neighbor.get_present_fe_values (); 

			//const unsigned int max_deg = std::max(fe_values_face.get_fe().tensor_degree() ,
				//fe_values_face_neighbor.get_fe().tensor_degree());

			////std::cout << cell->index() << "--" << face_no << "-->" << neighbor->index() << "( "  << neighbor2 <<" )"<< std::endl;

			////XXX: nieghbor2 dovrebbe essere l'indice della faccia, ma non sono sicuro
			//const double min_h = neighbor->face(neighbor2)->diameter();

			//const double penalty = 2*max_deg*(max_deg+1)/min_h;

			//face_term(cell, neighbor, fe_values_face, fe_values_face_neighbor, cell_matrix,
				    //aux_cell_neigh_matrix, aux_neigh_cell_matrix, aux_neigh_matrix,  penalty);

			//const unsigned int   neigh_dofs = neighbor->get_fe().dofs_per_cell;
			//neighbor_dof_indices.resize (neigh_dofs);
			//neighbor->get_dof_indices (neighbor_dof_indices);

			//cell_matrix.add(aux_cell_matrix,1);

			//constraints.distribute_local_to_global(aux_cell_neigh_matrix,
				//local_dof_indices,
				//neighbor_dof_indices,
				//system_matrix);
			//constraints.distribute_local_to_global(aux_neigh_cell_matrix,
				//neighbor_dof_indices,
				//local_dof_indices,
				//system_matrix);
			//constraints.distribute_local_to_global(aux_neigh_matrix,
				//neighbor_dof_indices,
				//neighbor_dof_indices,
				//system_matrix);


		    //}

		//}


	    //}
	//}



        local_dof_indices.resize (dofs_per_cell);
        cell->get_dof_indices (local_dof_indices);

        constraints.distribute_local_to_global (cell_matrix, cell_rhs_vec,
                                                local_dof_indices,
                                                system_matrix, system_rhs);
	//MatrixOut matrix_out;
	//std::ofstream out ("M.gnuplot");
	//matrix_out.build_patches (system_matrix, "M");
	//matrix_out.write_gnuplot (out);
      }
  }



  // @sect4{LaplaceProblem::solve}

  // The function solving the linear system is entirely unchanged from
  // previous examples. We simply try to reduce the initial residual (which
  // equals the $l_2$ norm of the right hand side) by a certain factor:
  template <int dim>
  void LaplaceProblem<dim>::solve ()
  {
    SolverControl           solver_control (system_rhs.size(),
                                            1e-10*system_rhs.l2_norm());
    SolverCG<>              cg (solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    cg.solve (system_matrix, solution, system_rhs,
              preconditioner);

    constraints.distribute (solution);
  }


  template <int dim>
  void LaplaceProblem<dim>::error()
  {
      const UpdateFlags update_flags = update_values
                                     | update_gradients
                                     | update_quadrature_points
				     | update_JxW_values;

      const UpdateFlags face_update_flags = update_values
	  | update_gradients
	  | update_quadrature_points
	  | update_JxW_values
	  | update_normal_vectors;

      //const UpdateFlags neighbor_face_update_flags = update_values;

      hp::FEValues<dim> hp_fe_values                   (mapping_collection , fe_collection , quadrature_collection      , update_flags);
      hp::FEFaceValues<dim> hp_fe_values_face          (mapping_collection , fe_collection , face_quadrature_collection , face_update_flags);
      hp::FESubfaceValues<dim> hp_fe_values_subface    (mapping_collection , fe_collection , face_quadrature_collection , face_update_flags);
      hp::FEFaceValues<dim> hp_fe_values_face_neighbor (mapping_collection , fe_collection , face_quadrature_collection , face_update_flags); //era neighbor_


      //const RightHandSide<dim> rhs_function;

      //FullMatrix<double>   cell_matrix;
      //FullMatrix<double>   aux_cell_matrix;
      //FullMatrix<double>   aux_cell_neigh_matrix;
      //FullMatrix<double>   aux_neigh_cell_matrix;
      //FullMatrix<double>   aux_neigh_matrix;
      //Vector<double>       cell_rhs;
      //Vector<double>       aux_cell_rhs;

      std::vector<types::global_dof_index> local_dof_indices;
      std::vector<types::global_dof_index> neighbor_dof_indices;
      L2_error = 0;

      //typename hp::DoFHandler<dim>::active_cell_iterator
      //cell = dof_handler.begin_active(),
      //endc = dof_handler.end();
      for (auto cell :  dof_handler.active_cell_iterators())
      {
	  const unsigned int   dofs_per_cell = cell->get_fe().dofs_per_cell;
	  local_dof_indices.resize (dofs_per_cell);
	  cell->get_dof_indices (local_dof_indices);

	  //cell_matrix.reinit (dofs_per_cell, dofs_per_cell);
	  //aux_cell_matrix.reinit (dofs_per_cell, dofs_per_cell);
	  //cell_matrix = 0;

	  //cell_rhs.reinit (dofs_per_cell);
	  //aux_cell_rhs.reinit (dofs_per_cell);
	  //cell_rhs = 0;

	  hp_fe_values.reinit (cell);
	  const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values ();

	  std::vector<double>  exact_values(fe_values.n_quadrature_points);
	  std::vector<double>  solution_values(fe_values.n_quadrature_points);
	  //std::vector<Tensor<1,dim> >   gradients()
	  exact_solution.value_list(fe_values.get_quadrature_points(), exact_values);
	  fe_values.get_function_values(solution, solution_values);

	  double cell_L2err = 0;
	  for(unsigned int k=0; k<fe_values.n_quadrature_points; ++k)
	  {
	      cell_L2err += (solution_values[k] - exact_values[k])*(solution_values[k] - exact_values[k])*fe_values.JxW(k);
	  }
	  L2_error += sqrt(cell_L2err);

      }



  }

  // @sect4{LaplaceProblem::postprocess}

  // After solving the linear system, we will want to postprocess the
  // solution. Here, all we do is to estimate the error, estimate the local
  // smoothness of the solution as described in the introduction, then write
  // graphical output, and finally refine the mesh in both $h$ and $p$
  // according to the indicators computed before. We do all this in the same
  // function because we want the estimated error and smoothness indicators
  // not only for refinement, but also include them in the graphical output.
  template <int dim>
  void LaplaceProblem<dim>::postprocess (const unsigned int cycle)
  {
    // Let us start with computing estimated error and smoothness indicators,
    // which each are one number for each active cell of our
    // triangulation. For the error indicator, we use the KellyErrorEstimator
    // class as always. Estimating the smoothness is done in the respective
    // function of this class; that function is discussed further down below:

    /*Vector<float> estimated_error_per_cell (triangulation.n_active_cells());*/
    //KellyErrorEstimator<dim>::estimate (dof_handler,
                                        //face_quadrature_collection,
                                        //typename FunctionMap<dim>::type(),
                                        //solution,
                                        //estimated_error_per_cell);


    //Vector<float> smoothness_indicators (triangulation.n_active_cells());
    //estimate_smoothness (smoothness_indicators);

    // Next we want to generate graphical output. In addition to the two
    // estimated quantities derived above, we would also like to output the
    // polynomial degree of the finite elements used on each of the elements
    // on the mesh.
    //
    // The way to do that requires that we loop over all cells and poll the
    // active finite element index of them using
    // <code>cell-@>active_fe_index()</code>. We then use the result of this
    // operation and query the finite element collection for the finite
    // element with that index, and finally determine the polynomial degree of
    // that element. The result we put into a vector with one element per
    // cell. The DataOut class requires this to be a vector of
    // <code>float</code> or <code>double</code>, even though our values are
    // all integers, so that it what we use:
    {
      Vector<float> fe_degrees (triangulation.n_active_cells());
      {
        typename hp::DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
        for (unsigned int index=0; cell!=endc; ++cell, ++index)
          fe_degrees(index)
            = fe_collection[cell->active_fe_index()].degree;
      }

      // With now all data vectors available -- solution, estimated errors and
      // smoothness indicators, and finite element degrees --, we create a
      // DataOut object for graphical output and attach all data. Note that
      // the DataOut class has a second template argument (which defaults to
      // DoFHandler@<dim@>, which is why we have never seen it in previous
      // tutorial programs) that indicates the type of DoF handler to be
      // used. Here, we have to use the hp::DoFHandler class:
      DataOut<dim,hp::DoFHandler<dim> > data_out;

      data_out.attach_dof_handler (dof_handler);
      data_out.add_data_vector (solution, "solution");
      //data_out.add_data_vector (estimated_error_per_cell, "error");
      //data_out.add_data_vector (smoothness_indicators, "smoothness");
      data_out.add_data_vector (fe_degrees, "fe_degree");
      data_out.build_patches ();

      // The final step in generating output is to determine a file name, open
      // the file, and write the data into it (here, we use VTK format):
      const std::string filename = "solution-" +
                                   Utilities::int_to_string (cycle, 2) +
                                   ".vtk";
      std::ofstream output (filename.c_str());
      data_out.write_vtk (output);
    }

    // After this, we would like to actually refine the mesh, in both $h$ and
    // $p$. The way we are going to do this is as follows: first, we use the
    // estimated error to flag those cells for refinement that have the
    // largest error. This is what we have always done:
    {
      //GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                       //estimated_error_per_cell,
                                                       //0.3, 0.03);

      //float max_smoothness = *std::min_element (smoothness_indicators.begin(),
                                                //smoothness_indicators.end()),
                             //min_smoothness = *std::max_element (smoothness_indicators.begin(),
                                                                 //smoothness_indicators.end());
      //{
        //typename hp::DoFHandler<dim>::active_cell_iterator
        //cell = dof_handler.begin_active(),
        //endc = dof_handler.end();
        //for (unsigned int index=0; cell!=endc; ++cell, ++index)
          //if (cell->refine_flag_set())
            //{
              //max_smoothness = std::max (max_smoothness,
                                         //smoothness_indicators(index));
              //min_smoothness = std::min (min_smoothness,
                                         //smoothness_indicators(index));
            //}
      //}
      //const float threshold_smoothness = (max_smoothness + min_smoothness) / 2;
      //{
        //typename hp::DoFHandler<dim>::active_cell_iterator
        //cell = dof_handler.begin_active(),
        //endc = dof_handler.end();
        //for (unsigned int index=0; cell!=endc; ++cell, ++index)
          //if (cell->refine_flag_set()
              //&&
              //(smoothness_indicators(index) > threshold_smoothness)
              //&&
              //(cell->active_fe_index()+1 < fe_collection.size()))
            //{
              //cell->clear_refine_flag();
              //cell->set_active_fe_index (cell->active_fe_index() + 1);
            //}
      //}

      //// At the end of this procedure, we then refine the mesh. During this
      //// process, children of cells undergoing bisection inherit their mother
      //// cell's finite element index:
    }
    for(auto cell : dof_handler.active_cell_iterators())
    {
	for (unsigned int v=0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
	{
	    //TODO:sostituire ocn singularity.distance :  potrei mettere la singolarita come public in exact solution
	    const double distance_from_singularity = exact_solution.get_singularity().distance (cell->vertex(v));
	    if (std::fabs(distance_from_singularity) < 1e-10)
	    {
		cell->set_refine_flag ();
		break;
	    }
	}
	cell->set_refine_flag ();
    }
    triangulation.execute_coarsening_and_refinement ();

/*    bool p_ref = true;*/
    //for(auto cell : dof_handler.active_cell_iterators())
    //{
	//p_ref = true;
	//for (unsigned int v=0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
	//{
	    //const double distance_from_singularity = exact_solution.get_singularity().distance (cell->vertex(v));
	    //if (std::fabs(distance_from_singularity) < 1e-10)
	    //{
		//p_ref = false;
		//break;
	    //}
	//}
	//if(p_ref && cell->active_fe_index() < fe_collection.size()-1)
	//{
	    //cell->set_active_fe_index(cell->active_fe_index() + 1);
	//}
    /*}*/
  }


  // @sect4{LaplaceProblem::create_coarse_grid}

  // The following function is used when creating the initial grid. It is a
  // specialization for the 2d case, i.e. a corresponding function needs to be
  // implemented if the program is run in anything other then 2d. The function
  // is actually stolen from step-14 and generates the same mesh used already
  // there, i.e. the square domain with the square hole in the middle. The
  // meaning of the different parts of this function are explained in the
  // documentation of step-14:
  template <int dim>
  void LaplaceProblem<dim>::create_coarse_grid ()
  {
    GridGenerator::subdivided_hyper_cube (triangulation, 2, -1., 1.);
  }




  // @sect4{LaplaceProblem::run}

  // This function implements the logic of the program, as did the respective
  // function in most of the previous programs already, see for example
  // step-6.
  //
  // Basically, it contains the adaptive loop: in the first iteration create a
  // coarse grid, and then set up the linear system, assemble it, solve, and
  // postprocess the solution including mesh refinement. Then start over
  // again. In the meantime, also output some information for those staring at
  // the screen trying to figure out what the program does:
  template <int dim>
  void LaplaceProblem<dim>::run ()
  {
    for (unsigned int cycle=0; cycle<5 ; ++cycle)
      {
        std::cout << "Cycle " << cycle << ':' << std::endl;

        if (cycle == 0)
          create_coarse_grid ();

        setup_system ();

        std::cout << "   Number of active cells:       "
                  << triangulation.n_active_cells()
                  << std::endl
                  << "   Number of degrees of freedom: "
                  << dof_handler.n_dofs()
                  << std::endl
                  << "   Number of constraints       : "
                  << constraints.n_constraints()
                  << std::endl;

        assemble_system ();
        solve ();
	error ();
	std::cout << "L2 error: " << dof_handler.n_dofs() << " " << L2_error << std::endl; 
	postprocess (cycle);
      }
  }


  // @sect4{LaplaceProblem::estimate_smoothness}

  // This last function of significance implements the algorithm to estimate
  // the smoothness exponent using the algorithms explained in detail in the
  // introduction. We will therefore only comment on those points that are of
  // implementational importance.
  template <int dim>
  void
  LaplaceProblem<dim>::
  estimate_smoothness (Vector<float> &smoothness_indicators) const
  {
    // The first thing we need to do is to define the Fourier vectors ${\bf
    // k}$ for which we want to compute Fourier coefficients of the solution
    // on each cell. In 2d, we pick those vectors ${\bf k}=(\pi i, \pi j)^T$
    // for which $\sqrt{i^2+j^2}\le N$, with $i,j$ integers and $N$ being the
    // maximal polynomial degree we use for the finite elements in this
    // program. The 3d case is handled analogously. 1d and dimensions higher
    // than 3 are not implemented, and we guard our implementation by making
    // sure that we receive an exception in case someone tries to compile the
    // program for any of these dimensions.
    //
    // We exclude ${\bf k}=0$ to avoid problems computing $|{\bf k}|^{-mu}$
    // and $\ln |{\bf k}|$. The other vectors are stored in the field
    // <code>k_vectors</code>. In addition, we store the square of the
    // magnitude of each of these vectors (up to a factor $\pi^2$) in the
    // <code>k_vectors_magnitude</code> array -- we will need that when we
    // attempt to find out which of those Fourier coefficients corresponding
    // to Fourier vectors of the same magnitude is the largest:
    const unsigned int N = max_degree;

    std::vector<Tensor<1,dim> > k_vectors;
    std::vector<unsigned int>   k_vectors_magnitude;
    switch (dim)
      {
      case 2:
      {
        for (unsigned int i=0; i<N; ++i)
          for (unsigned int j=0; j<N; ++j)
            if (!((i==0) && (j==0))
                &&
                (i*i + j*j < N*N))
              {
                k_vectors.push_back (Point<dim>(numbers::PI * i,
                                                numbers::PI * j));
                k_vectors_magnitude.push_back (i*i+j*j);
              }

        break;
      }

      case 3:
      {
        for (unsigned int i=0; i<N; ++i)
          for (unsigned int j=0; j<N; ++j)
            for (unsigned int k=0; k<N; ++k)
              if (!((i==0) && (j==0) && (k==0))
                  &&
                  (i*i + j*j + k*k < N*N))
                {
                  k_vectors.push_back (Point<dim>(numbers::PI * i,
                                                  numbers::PI * j,
                                                  numbers::PI * k));
                  k_vectors_magnitude.push_back (i*i+j*j+k*k);
                }

        break;
      }

      default:
        Assert (false, ExcNotImplemented());
      }

    // After we have set up the Fourier vectors, we also store their total
    // number for simplicity, and compute the logarithm of the magnitude of
    // each of these vectors since we will need it many times over further
    // down below:
    const unsigned n_fourier_modes = k_vectors.size();
    std::vector<double> ln_k (n_fourier_modes);
    for (unsigned int i=0; i<n_fourier_modes; ++i)
      ln_k[i] = std::log (k_vectors[i].norm());


    // Next, we need to assemble the matrices that do the Fourier transforms
    // for each of the finite elements we deal with, i.e. the matrices ${\cal
    // F}_{{\bf k},j}$ defined in the introduction. We have to do that for
    // each of the finite elements in use. Note that these matrices are
    // complex-valued, so we can't use the FullMatrix class. Instead, we use
    // the Table class template.
    std::vector<Table<2,std::complex<double> > >
    fourier_transform_matrices (fe_collection.size());

    // In order to compute them, we of course can't perform the Fourier
    // transform analytically, but have to approximate it using quadrature. To
    // this end, we use a quadrature formula that is obtained by iterating a
    // 2-point Gauss formula as many times as the maximal exponent we use for
    // the term $e^{i{\bf k}\cdot{\bf x}}$:
    QGauss<1>      base_quadrature (2);
    QIterated<dim> quadrature (base_quadrature, N);

    // With this, we then loop over all finite elements in use, reinitialize
    // the respective matrix ${\cal F}$ to the right size, and integrate each
    // entry of the matrix numerically as ${\cal F}_{{\bf k},j}=\sum_q
    // e^{i{\bf k}\cdot {\bf x}}\varphi_j({\bf x}_q) w_q$, where $x_q$ are the
    // quadrature points and $w_q$ are the quadrature weights. Note that the
    // imaginary unit $i=\sqrt{-1}$ is obtained from the standard C++ classes
    // using <code>std::complex@<double@>(0,1)</code>.

    // Because we work on the unit cell, we can do all this work without a
    // mapping from reference to real cell and consequently do not need the
    // FEValues class.
    for (unsigned int fe=0; fe<fe_collection.size(); ++fe)
      {
        fourier_transform_matrices[fe].reinit (n_fourier_modes,
                                               fe_collection[fe].dofs_per_cell);

        for (unsigned int k=0; k<n_fourier_modes; ++k)
          for (unsigned int j=0; j<fe_collection[fe].dofs_per_cell; ++j)
            {
              std::complex<double> sum = 0;
              for (unsigned int q=0; q<quadrature.size(); ++q)
                {
                  const Point<dim> x_q = quadrature.point(q);
                  sum += std::exp(std::complex<double>(0,1) *
                                  (k_vectors[k] * x_q)) *
                         fe_collection[fe].shape_value(j,x_q) *
                         quadrature.weight(q);
                }
              fourier_transform_matrices[fe](k,j)
                = sum / std::pow(2*numbers::PI, 1.*dim/2);
            }
      }

    // The next thing is to loop over all cells and do our work there, i.e. to
    // locally do the Fourier transform and estimate the decay coefficient. We
    // will use the following two arrays as scratch arrays in the loop and
    // allocate them here to avoid repeated memory allocations:
    std::vector<std::complex<double> > fourier_coefficients (n_fourier_modes);
    Vector<double>                     local_dof_values;

    // Then here is the loop:
    typename hp::DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (unsigned int index=0; cell!=endc; ++cell, ++index)
      {
        // Inside the loop, we first need to get the values of the local
        // degrees of freedom (which we put into the
        // <code>local_dof_values</code> array after setting it to the right
        // size) and then need to compute the Fourier transform by multiplying
        // this vector with the matrix ${\cal F}$ corresponding to this finite
        // element. We need to write out the multiplication by hand because
        // the objects holding the data do not have <code>vmult</code>-like
        // functions declared:
        local_dof_values.reinit (cell->get_fe().dofs_per_cell);
        cell->get_dof_values (solution, local_dof_values);

        for (unsigned int f=0; f<n_fourier_modes; ++f)
          {
            fourier_coefficients[f] = 0;

            for (unsigned int i=0; i<cell->get_fe().dofs_per_cell; ++i)
              fourier_coefficients[f] +=
                fourier_transform_matrices[cell->active_fe_index()](f,i)
                *
                local_dof_values(i);
          }

        // The next thing, as explained in the introduction, is that we wanted
        // to only fit our exponential decay of Fourier coefficients to the
        // largest coefficients for each possible value of $|{\bf k}|$. To
        // this end, we create a map that for each magnitude $|{\bf k}|$
        // stores the largest $|\hat U_{{\bf k}}|$ found so far, i.e. we
        // overwrite the existing value (or add it to the map) if no value for
        // the current $|{\bf k}|$ exists yet, or if the current value is
        // larger than the previously stored one:
        std::map<unsigned int, double> k_to_max_U_map;
        for (unsigned int f=0; f<n_fourier_modes; ++f)
          if ((k_to_max_U_map.find (k_vectors_magnitude[f]) ==
               k_to_max_U_map.end())
              ||
              (k_to_max_U_map[k_vectors_magnitude[f]] <
               std::abs (fourier_coefficients[f])))
            k_to_max_U_map[k_vectors_magnitude[f]]
              = std::abs (fourier_coefficients[f]);
        // Note that it comes in handy here that we have stored the magnitudes
        // of vectors as integers, since this way we do not have to deal with
        // round-off-sized differences between different values of $|{\bf
        // k}|$.

        // As the final task, we have to calculate the various contributions
        // to the formula for $\mu$. We'll only take those Fourier
        // coefficients with the largest magnitude for a given value of $|{\bf
        // k}|$ as explained above:
        double  sum_1           = 0,
                sum_ln_k        = 0,
                sum_ln_k_square = 0,
                sum_ln_U        = 0,
                sum_ln_U_ln_k   = 0;
        for (unsigned int f=0; f<n_fourier_modes; ++f)
          if (k_to_max_U_map[k_vectors_magnitude[f]] ==
              std::abs (fourier_coefficients[f]))
            {
              sum_1 += 1;
              sum_ln_k += ln_k[f];
              sum_ln_k_square += ln_k[f]*ln_k[f];
              sum_ln_U += std::log (std::abs (fourier_coefficients[f]));
              sum_ln_U_ln_k += std::log (std::abs (fourier_coefficients[f])) *
                               ln_k[f];
            }

        // With these so-computed sums, we can now evaluate the formula for
        // $\mu$ derived in the introduction:
        const double mu
          = (1./(sum_1*sum_ln_k_square - sum_ln_k*sum_ln_k)
             *
             (sum_ln_k*sum_ln_U - sum_1*sum_ln_U_ln_k));

        // The final step is to compute the Sobolev index $s=\mu-\frac d2$ and
        // store it in the vector of estimated values for each cell:
        smoothness_indicators(index) = mu - 1.*dim/2;
      }
  }
}


// @sect3{The main function}

// The main function is again verbatim what we had before: wrap creating and
// running an object of the main class into a <code>try</code> block and catch
// whatever exceptions are thrown, thereby producing meaningful output if
// anything should go wrong:
int main ()
{
  try
    {
      using namespace dealii;
      using namespace Step27;

      deallog.depth_console (0);

      LaplaceProblem<2> laplace_problem;
      laplace_problem.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
