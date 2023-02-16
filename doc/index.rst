.. project-template documentation master file, created by
   sphinx-quickstart on Mon Jan 18 14:44:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to VolterraBasis's documentation!
============================================

This project compute position-dependent memory kernel for Generalized Langevin Equations. Please refer to  Position-dependent memory kernel in generalized Langevin equations: Theory and numerical estimation, J. Chem. Phys. 156, 244105 (2022); https://doi.org/10.1063/5.0094566, also available at https://arxiv.org/abs/2201.02457 for a detailed description of the algorithm.


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   self
   quick_start
   api


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorial - Examples

   auto_examples/index



Installation
------------------

Run
    >>> pip install git+https://github.com/HadrienNU/VolterraBasis.git
    
to install.

Getting Started
------------------

To run the code, you should first instanciate the Estimator_gle class


    >>> mymem =  Estimator_gle(traj_list, vb.Pos_gle, bf.BSplineFeatures(Nsplines))

The mandatory arguments are a list of trajectories, the choice of a model and a funcionnal basis.

The list of trajectories should be created through the :meth:`VolterraBasis.xframe` method such as

    >>> trj = np.loadtxt("example_lj.trj")
    >>> xf = vb.xframe(trj[:, 1], trj[:, 0]) # First argument is trajectory, second is time
    >>> xvaf = vb.compute_va(xf) #  Compute velocity and acceleration
    >>> trajs_list=[xvaf]

You should then compute mean force and correlation using

    >>> mymem.compute_mean_force()
    >>> mymem.compute_corrs()

Inversion of Volterra Integral Equations
------------------------------------------------------

Computation of the memory kernel is obtained using

    >>> mymem.compute_kernel()

Several algorithms for the inversion of the Volterra Integral Equations are available. Please refer to P. Linz, “Numerical methods for Volterra integral equations of the first kind”, The Computer
Journal 12, 393–397 (1969) for mathematical details.

Functionnal basis
------------------

The estimation of the memory kernel necessite the choice of a functionnal basis. Functional basis are implemented in :class:`VolterraBasis.basis` that could be imported and initialized as ::

    >>> import VolterraBasis.basis as bf
    >>> basis=bf.BSplineFeatures(15)

Several options are available for the type of basis, please refer to the documentation. Although multidimensionnal trajectories can be analysed, not all functionnal basis are multidimensionnal.


Force and memory estimate
-------------------------


Once the mean force and memory have been computed, the value of the force and memory kernel at given position can be computed trought function :meth:`VolterraBasis.Pos_gle.force_eval` and :meth:`VolterraBasis.Pos_gle.kernel_eval`

Choice of the form of the GLE
-----------------------------

Several options are available to choose the form of the GLE:

* :class:`VolterraBasis.Pos_gle` implement the form of the GLE featured in Vroylandt and Monmarché with memory kernel linear in velocity.
* :class:`VolterraBasis.Pos_gle_with_friction` is similar to the previous but don't assume that the instantaneous friction is zero.
* :class:`VolterraBasis.Pos_gle_const_kernel`  is the traditionnal GLE with memory kernel linear in velocity and independant of position.
* :class:`VolterraBasis.Pos_gle_no_vel_basis`  implement a GLE where the memory kernel has no dependance in velocity.
* :class:`VolterraBasis.Pos_gle_overdamped` compute the memory kernel for an overdamped dynamics.
