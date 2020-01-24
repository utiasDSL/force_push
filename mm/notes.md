# Sim Notes

* Hybrid vs. parallel control logic
  - one idea is that we could do both, where we constrain certain directions to
    be force controlled at certain times (e.g. when I expect to encounter a
    surface)
  - wonder if it would be worth thinking about logic to prevent aggressive
    motion in the case that no surface is actually present
* Need some state machine logic to accomplish more complex tasks that aren't
  just complying to disturbances

* Recall whether or not to use the discharge term (i.e. this is task-dependent)
  - use in order to return to nominal desired trajectory
  - don't use if you want the EE to be repositioned by interaction

* Scenarios
  - complex motion control without orientation control (e.g. spiral)
  - motion control with orientation control
  - responding to objects that apply force
  - responding to objects that apply force and doing orientation control such
    that the EE is normal to the applied force

* High force damper coefficient (b=1000, with k=10000) led to instability
* With smaller dt (i.e. approaching continuous time), there was an offset from
  the freespace desired position and further penetration into the obstacle
* Compelling case for the discharge term in the scenario with circular
  trajectory and circular obstacle
  - bahaviour is nonsensical with no decay

* Problem: exerted force is way higher than desired force
