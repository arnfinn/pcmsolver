/*! \file WEMSolver.h 
\brief PCM solver
*/


#ifndef WEMSOLVER_H_
#define WEMSOLVER_H_

#include <string>
#include <vector>
#include <iostream>
#include <complex>

extern "C"{
#include "vector3.h"
#include "sparse2.h"
}

using namespace std;

template<class T>
class WEMSolver : public PCMSolver<T> {
 public:
    WEMSolver(GreensFunction<T> &gfi, GreensFunction<T> &gfo);
    WEMSolver(GreensFunction<T> *gfi, GreensFunction<T> *gfo);
    WEMSolver(Section solver);
    ~WEMSolver();
    vector3 **** getT_(){return T_;}
    int getQuadratureLevel(){return quadratureLevel_;}
    virtual void buildSystemMatrix(Cavity & cavity);
    virtual void constructSystemMatrix() = 0;
    virtual void initInterpolation() = 0;
    virtual void uploadCavity(WaveletCavity cavity); // different interpolation
    virtual VectorXd compCharge(const VectorXd & potential, VectorXd & charge) = 0;
    virtual VectorXd compCharge(const VectorXd & potential);
    double SL(vector3 x, vector3 y);
    double DL(vector3 x, vector3 y, vector3 n_y);
    void fixPointersInside();
    void fixPointersOutside();
 protected:
    GreensFunction<T> * gf;
    double threshold;
    unsigned int quadratureLevel_;
    sparse2 S_i_, S_e_; // System matrices
    bool systemMatricesInitialized_;
    vector3 *** pointList; // the old U
    vector3 *nodeList; //*P_; --     // Point list
    unsigned int **elementList; //**F_;     // Element list
    vector3 ****T_; //--    // Something 1 
    unsigned int nNodes; //np_; --    // Number of knot points or something
    unsigned int nFunctions; //nf_; --    // Number of ansatz functions
    unsigned int nPatches; // p_; --    // Number of points 
    unsigned int nLevels; //M_; --    // Patch level (2**M * 2**M elements per patch)
    int nQuadPoints; // nPoints_;    // Number of quadrature points
    double apriori1_, aposteriori1_;    // System matrix sparsities
    double apriori2_, aposteriori2_;    // System matrix sparsities
};
#endif
