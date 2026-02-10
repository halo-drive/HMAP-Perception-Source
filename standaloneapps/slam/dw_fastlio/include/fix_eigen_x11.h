/////////////////////////////////////////////////////////////////////////////////////////
// Fix for Eigen/X11 conflict
// X11 defines Success as 0, which conflicts with Eigen's use of Success
// This header undefines Success before Eigen headers are included
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef FIX_EIGEN_X11_H_
#define FIX_EIGEN_X11_H_

#ifdef Success
#undef Success
#endif

#endif // FIX_EIGEN_X11_H_

