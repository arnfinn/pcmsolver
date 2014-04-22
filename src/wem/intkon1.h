/* warning-disabler-start */

#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wextra"
#elif defined(__ICC) || defined(__INTEL_COMPILER)
#pragma warning push
#pragma warning disable "-Wall"
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wall"
#pragma clang diagnostic ignored "-Weffc++"
#pragma clang diagnostic ignored "-Wextra"
#pragma clang diagnostic ignored "-Wdocumentation"
#pragma clang diagnostic ignored "-Wdeprecated-register"
#pragma clang diagnostic ignored "-Wincompatible-pointer-types"
#pragma clang diagnostic ignored "-Wempty-body"
#endif

/* warning-disabler-end */

#ifndef INTKON1_H_
#define INTKON1_H_
/***************
 *  IntKon1.h  *
 ***************/


/*===========================================================*
 *  Fernfeld-Quadratur-Routine:			             *
 *  Die in der Funktion init_randwerte vorab berechneten     *
 *  Auswertepunkte und Gewichte der Gauss-Quadratur werden   *
 *  in IntKon1 zum entsprechenden Integral zusammengefuegt.  *
 *===========================================================*/


#include "randwerte.h"

void IntKon1(double *c, element *element1, element *element2, randwerte *RW, cubature *Q1, cubature *Q2, vector3 ****P, unsigned int M, double (*SL) (), double (*DL) ());
/* No-Problem-Quadrature-Routine */
#endif
/* warning-disabler-start */

#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#pragma GCC diagnostic pop
#elif defined(__ICC) || defined(__INTEL_COMPILER)
#pragma warning pop
#elif defined(__clang__)
#pragma clang diagnostic pop
#endif

/* warning-disabler-end */

