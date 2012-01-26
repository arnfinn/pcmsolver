/*! \file GreensFunction.h 
\brief Abstract base class for the Green´s function generator.
*/

#ifndef GREENSFUNCTION
#define GREENSFUNCTION

/** Green´s function Abstract base class 

A generic green´s function to reprensent the electrostatic potential for a given environment

*/

class Section;

template<class T>
class GreensFunction
{
 public:
    GreensFunction(){delta = 1.0e-4;}
    virtual ~GreensFunction(){};
    virtual double evalf(Vector3d &p1, Vector3d &p2);
    virtual double evald(Vector3d &direction, Vector3d &p1, Vector3d &p2) = 0;
    virtual double derivativeSource(Vector3d &direction, Vector3d &p1, Vector3d &p2);
    virtual double derivativeProbe(Vector3d &direction, Vector3d &p1, Vector3d &p2);
    virtual Vector3d gradientSource(Vector3d &p1, Vector3d &p2);
    virtual Vector3d gradientProbe(Vector3d &p1, Vector3d &p2);
    virtual void gradientSource(Vector3d &gradient, Vector3d &p1, Vector3d &p2);
    virtual void gradientProbe(Vector3d &gradient, Vector3d &p1, Vector3d &p2);
    void setDelta(double value);
    double getDelta(){return delta;}
    bool isUniform(){ return uniformFlag; }
    GreensFunction<T> * allocateGreensFunction(const Section &green);
    GreensFunction<T> * allocateGreensFunction(double dielConst);
    GreensFunction<T> * allocateGreensFunction();
    friend std::ostream& operator<<(std::ostream &os, GreensFunction<T> &gf) {
        return gf.printObject(os);
    };
 protected:
    virtual T evalGreensFunction(T * source, T * probe) = 0;
    virtual std::ostream & printObject(std::ostream & os);
    double delta;
	bool uniformFlag;
};


#endif
