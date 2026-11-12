Magnetohydrodynamic Rayleigh-Bénard Convection (RBC-MHD)
========================================================

RBC-MHD models Rayleigh-Bénard convection of a liquid metal in a horizontal magnetic
field :math:`\mathbf{B}`. A box is heated from below and cooled from above, and the
field points along :math:`x`, normal to the two Hartmann walls. The Lorentz force
damps motion across the field lines. As a result, the flow organizes into convection
rolls aligned with :math:`\mathbf{B}`, whose number and stability depend on how strong
buoyancy is compared with the magnetic field [1]_.

The system is governed by three dimensionless parameters:

- the Rayleigh number :math:`\mathrm{Ra}`, which sets the strength of buoyancy;
- the Chandrasekhar number :math:`Q = \mathrm{Ha}^2`, which sets the strength of the
  magnetic field (:math:`\mathrm{Ha}` is the Hartmann number);
- the Prandtl number :math:`\mathrm{Pr}`, a material property of the fluid.

The Lorentz force relative to inertia is measured by the Stuart number
:math:`N = Q \sqrt{\mathrm{Pr} / \mathrm{Ra}}`.

All environments use :math:`Q = 10^3` (:math:`\mathrm{Ha} \approx 31.6`),
:math:`\mathrm{Pr} = 0.025` (a liquid metal) and a box of
:math:`5 \times 1 \times 5` in units of the height :math:`H`. The grid is refined
towards the Hartmann walls, so that the thin Hartmann layers are resolved by at least
``min_layer_cells`` cells.

Environment List
----------------

+-----------------------------+------------+------------+--------------------------+
| Environment ID              | Rayleigh   | Resolution | Notes                    |
+=============================+============+============+==========================+
| ``RBCMHD3D-easy-v0``        | 2×10⁴      | 32         | 20 electrode pairs       |
+-----------------------------+------------+------------+--------------------------+
| ``RBCMHD3D-medium-v0``      | 5×10⁴      | 32         | 20 electrode pairs       |
+-----------------------------+------------+------------+--------------------------+
| ``RBCMHD3D-hard-v0``        | 1×10⁵      | 48         | 20 electrode pairs       |
+-----------------------------+------------+------------+--------------------------+

The resolution is the number of cells along the vertical axis :math:`y`. The
:math:`x` and :math:`z` axes have ``round(aspect_ratio * resolution)`` cells.

Reward
------

Unlike :doc:`/api/rbc`, the objective is to *increase* convective heat transfer, at a
small cost in the electrical power needed for actuation. The volume-averaged Nusselt
number is

.. math::

   \mathrm{Nu} = 1 + \sqrt{\mathrm{Ra} \, \mathrm{Pr}} \,
   \langle u_y \, (T - T_{\mathrm{ref}}) \rangle_V

where :math:`u_y` is the vertical velocity, :math:`T` the temperature,
:math:`T_{\mathrm{ref}}` the mean plate temperature, and :math:`\langle \cdot \rangle_V`
the volume average. The reward is

.. math::

   r_t = \frac{\mathrm{Nu}}{\mathrm{Nu}_{\mathrm{ref}}} - 1
   - w_P \, \frac{P_{\mathrm{el}}}{P_{\mathrm{ref}}}

Here :math:`P_{\mathrm{el}}` is the electrical power the electrodes feed into the fluid
and :math:`w_P` is ``power_penalty_weight`` (default 0.01). Both terms are averaged over
the simulation steps of one environment step. :math:`\mathrm{Nu}_{\mathrm{ref}}` and
:math:`P_{\mathrm{ref}}` are the mean Nusselt number and the mean buoyancy power of the
uncontrolled flow. Joule heating in the Hartmann layers costs much more power than
buoyancy provides, which is why :math:`w_P` is small.

In MARL mode, each agent's reward combines two parts. The first is
``local_reward_weight`` times the same Nusselt term, evaluated over the columns of the
agent's observation window. The second is the rest of the global reward, as in
:class:`~fluidgym.envs.rbc.RBCEnv3D`.

Action Space
------------

The flow is controlled by electrodes in the two Hartmann walls. Each wall is divided
into ``n_electrodes`` patches along :math:`y` (default 2) and ``n_stations`` stations
along :math:`z`. The default of ``n_stations`` is ``round(aspect_ratio * n_electrodes)``,
which gives 10. Each patch holds an electrode with a raised-cosine profile. It covers
``electrode_fill`` (default 0.5) of the patch pitch in :math:`y` and :math:`z`, and
insulating wall surrounds it. Electrode :math:`k` exists on both walls. The pair
carries the current :math:`I_k`, half of it through each wall.

The actions in :math:`[-1, 1]` set the currents :math:`I_k`. They are:

- centred to zero net current, which the insulating box requires,
- clipped to the action bounds and centred again,
- scaled by ``max_current`` (in units of :math:`\sigma U B H^2`).

- **SARL** (default): one action of shape ``(n_stations, n_electrodes)``, i.e.
  ``(10, 2)``.
- **MARL**: each station along :math:`z` is an agent that sets the currents of its
  column of electrodes, giving 10 agents with actions of shape ``(n_electrodes,)``.

Observation Space
-----------------

Observations are a dictionary with the ``temperature`` and the three ``velocity``
components at a regular grid of sensors. There are 2 sensors per electrode station
along :math:`z` and 8 sensors along :math:`y`. The :math:`x` axis gets the same sensor
spacing as :math:`z`.

- **SARL**: all sensors. The temperature has shape ``(20, 8, 20)`` and the velocity
  ``(3, 20, 8, 20)`` (ordered :math:`z, y, x`).
- **MARL**: each agent observes a window of ``local_obs_window`` (default 3) stations
  centred on its own, i.e. ``(6, 8, 20)``. Beyond the side walls, the window is padded
  with the quiescent conduction state.

Difficulty Levels
-----------------

Difficulty is set by the Rayleigh number at fixed :math:`Q = 10^3`, so the Hartmann
layers keep the same thickness:

+------------+--------------------+------------------------------------------+
| Level      | Rayleigh           | Regime [1]_                              |
+============+====================+==========================================+
| Easy       | Ra = 2×10⁴ (20 Q)  | Steady rolls                             |
+------------+--------------------+------------------------------------------+
| Medium     | Ra = 5×10⁴ (50 Q)  | Transition from 4 to 3 rolls             |
+------------+--------------------+------------------------------------------+
| Hard       | Ra = 1×10⁵ (100 Q) | Boundary of the no-roll regime           |
+------------+--------------------+------------------------------------------+

At higher Rayleigh numbers the flow is more chaotic, and the Lorentz force acting on it
is weaker (lower :math:`N`). The hard level uses a finer grid.

References
----------

.. [1] T. Yanagisawa, Y. Hamano, T. Miyagoshi, Y. Yamagishi, Y. Tasaka, and Y. Takeda,
   "Convection patterns in a liquid metal under an imposed horizontal magnetic field,"
   *Phys. Rev. E*, vol. 88, no. 6, p. 063020, 2013, doi: 10.1103/PhysRevE.88.063020.

API Reference
-------------

.. autosummary::
   :toctree: generated/

   fluidgym.envs.rbc_mhd.RBCMHDEnv3D
