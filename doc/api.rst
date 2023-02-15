####################
VolterraBasis API
####################

.. currentmodule:: VolterraBasis


Loading trajectories
=========================
.. autosummary::
   :toctree: generated/
   :template: function.rst

   xframe

   compute_va

   compute_a

 Memory kernel estimation
 =========================

 .. autosummary::
    :toctree: generated/
    :template: class.rst

    Estimator_gle

 Available models of GLE
=========================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Pos_gle

   Pos_gle_with_friction

   Pos_gle_no_vel_basis

   Pos_gle_const_kernel

   Pos_gle_overdamped

   Pos_gle_hybrid

Basis Features
===============

.. autosummary::
   :toctree: generated/
   :template: class.rst

   VolterraBasis.basis.LinearFeatures

   VolterraBasis.basis.PolynomialFeatures

   VolterraBasis.basis.FourierFeatures

   VolterraBasis.basis.BSplineFeatures

   VolterraBasis.basis.FEMScalarFeatures

   VolterraBasis.basis.SmoothIndicatorFeatures

   VolterraBasis.basis.SplineFctFeatures

   VolterraBasis.basis.FeaturesCombiner

   VolterraBasis.basis.TensorialBasis2D
