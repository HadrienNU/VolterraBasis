###########
Quick Start
###########

This package aims at computing memory kernel when studying Generalized Langevin Equations (GLE).

Inversion of Volterra Integral Equations
===================================================



Several algorithms for the inversion of the Volterra Integral Equations are available. Please refer to Linz for mathematical details.

Functionnal basis
------------------

The estimation of the memory kernel necessite the choice of a functionnal basis. Functional basis are implemented in :class:`VolterraBasis.basis` that could be imported and initialized as ::

    >>> import VolterraBasis.basis as bf
    >>> basis=bf.BSplineFeatures(15)

Several options are available for the type of basis, please refer to the documentation of  :class:`VolterraBasis.VolterraBasis`. 


Force and memory estimate
-------------------------


Once the mean force and memory have been computed, the value of the force and memory kernel at given position can be computed trought function :meth:`VolterraBasis.Pos_gle.dU` and :meth:`VolterraBasis.Pos_gle.kernel_eval`

Choice of the form of the GLE
-----------------------------

Several options are available to choose the form of the GLE:
* :class:`VolterraBasis.Pos_gle` implement the form of the GLE featured in Vroylandt and Monmarché with memory kernel linear in velocity.
* :class:`VolterraBasis.Pos_gle_with_friction` is similar to the previous but don't assume that the instantaneous friction is zero.
* :class:`VolterraBasis.Pos_gle_no_vel_basis`  implement a GLE where the memory kernel has no dependance in velocity.
* :class:`VolterraBasis.Pos_gle_overdamped` compute the memory kernel for an overdamped dynamics.


