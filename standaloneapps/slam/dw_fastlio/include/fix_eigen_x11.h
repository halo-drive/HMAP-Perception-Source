/////////////////////////////////////////////////////////////////////////////////////////
// Fix for Eigen/X11 conflict
// X11 defines Success as 0, which conflicts with Eigen's use of Success
// This header undefines Success before Eigen headers are included
/////////////////////////////////////////////////////////////////////////////////////////

// #ifndef FIX_EIGEN_X11_H_
// #define FIX_EIGEN_X11_H_

// #ifdef Success
// #undef Success
// #endif

// #endif // FIX_EIGEN_X11_H_

#ifndef FIX_EIGEN_X11_H_
#define FIX_EIGEN_X11_H_

#if __has_include(<X11/Xlib.h>)
#include <X11/Xlib.h>
#endif

#if __has_include(<X11/Xlib.h>)
#include <X11/Xlib.h>
#endif

#ifdef Success
#undef Success
#endif

#ifdef Status
#undef Status
#endif

#endif // FIX_EIGEN_X11_H_