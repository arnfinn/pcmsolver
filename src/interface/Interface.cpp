/* pcmsolver_copyright_start */
/*
 *     PCMSolver, an API for the Polarizable Continuum Model
 *     Copyright (C) 2013 Roberto Di Remigio, Luca Frediani and contributors
 *
 *     This file is part of PCMSolver.
 *
 *     PCMSolver is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU Lesser General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 *
 *     PCMSolver is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU Lesser General Public License for more details.
 *
 *     You should have received a copy of the GNU Lesser General Public License
 *     along with PCMSolver.  If not, see <http://www.gnu.org/licenses/>.
 *
 *     For information on the complete list of contributors to the
 *     PCMSolver API, see: <http://pcmsolver.github.io/pcmsolver-doc>
 */
/* pcmsolver_copyright_end */

/*

  Interface functions implementation

*/
#include "Interface.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "Config.hpp"

#include <Eigen/Core>

// Include Boost headers here
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

// Core classes
// This list all header files that need to be included here.
// It is automatically generated by CMake during configuration
#include "Includer.hpp"

typedef std::map<std::string, SurfaceFunction> SurfaceFunctionMap;

// We need globals as they must be accessible across all the functions defined in this interface...
// The final objective is to have only a pointer to Cavity and a pointer to PCMSolver (our abstractions)
// then maybe manage them through "objectification" of this interface.
SharedCavity _cavity;
SharedPCMSolver _solver;
SharedPCMSolver _noneqSolver;
bool noneqExists = false;

SurfaceFunctionMap functions;
SharedInput parsedInput;
std::vector<std::string> input_strings; // Used only by Fortran hosts

/*

	Functions visible to host program

*/

#ifdef __cplusplus
extern "C" {
#endif

void hello_pcm(int * a, double * b)
{
    std::ostringstream out_stream;
    out_stream << "Hello, PCM!" << std::endl;
    out_stream << "The integer is: " << *a << std::endl;
    out_stream << "The double is: " << *b << std::endl;
    printer(out_stream);
}

void set_up_pcm(int * host_provides_input)
{
    bool from_host = false;
    if (*host_provides_input != 0) {
	    from_host = true;
    }
    setupInput(from_host);
    initCavity();
    initSolver();
}

void tear_down_pcm()
{
    // Delete all the global pointers, maybe in a more refined way...
    functions.clear();
}

void write_timings()
{
    // Print out timings to pcmsolver.timer.dat
    TIMER_DONE("pcmsolver.timer.dat");
}

void compute_asc(char * potString, char * chgString, int * irrep)
{
    std::string potFuncName(potString);
    std::string chgFuncName(chgString);

    // Get the proper iterators
    SurfaceFunctionMap::const_iterator iter_pot = functions.find(potFuncName);
    // Here we check whether the function exists already or not
    // 1. find the lower bound of the map
    SurfaceFunctionMap::iterator iter_chg = functions.lower_bound(chgFuncName);
    // 2. if iter_chg == end, or if iter_chg is not a match,
    //    then this element was not in the map, so we need to insert it
    if ( iter_chg == functions.end()  ||  iter_chg->first != chgFuncName ) {
        // move iter_chg to the element preceeding the insertion point
        if ( iter_chg != functions.begin() ) --iter_chg;
        // insert it
	    SurfaceFunction func(chgFuncName, _cavity->size());
        auto insertion = SurfaceFunctionMap::value_type(chgFuncName, func);
        iter_chg = functions.insert(iter_chg, insertion);
    }

    // We clear the ASC surface function. Needed when using symmetry for response calculations
    iter_chg->second.clear();
    iter_chg->second.vector() = _solver->computeCharge(iter_pot->second.vector(), *irrep);
    // Renormalization of charges: divide by the number of symmetry operations in the group
    (iter_chg->second) /= double(_cavity->pointGroup().nrIrrep());
}

void compute_nonequilibrium_asc(char * potString, char * chgString, int * irrep)
{
    // Check that the nonequilibrium solver has been created
    if (!noneqExists) {
        initNonEqSolver();
    }

    std::string potFuncName(potString);
    std::string chgFuncName(chgString);

    // Get the proper iterators
    SurfaceFunctionMap::const_iterator iter_pot = functions.find(potFuncName);
    // Here we check whether the function exists already or not
    // 1. find the lower bound of the map
    SurfaceFunctionMap::iterator iter_chg = functions.lower_bound(chgFuncName);
    // 2. if iter_chg == end, or if iter_chg is not a match,
    //    then this element was not in the map, so we need to insert it
    if ( iter_chg == functions.end()  ||  iter_chg->first != chgFuncName ) {
        // move iter_chg to the element preceeding the insertion point
        if ( iter_chg != functions.begin() ) --iter_chg;
        // insert it
	    SurfaceFunction func(chgFuncName, _cavity->size());
        auto insertion = SurfaceFunctionMap::value_type(chgFuncName, func);
        iter_chg = functions.insert(iter_chg, insertion);
    }

    // If it already exists there's no problem, we will pass a reference to its values to
    iter_chg->second.vector() = _noneqSolver->computeCharge(iter_pot->second.vector(), *irrep);
    // Renormalization of charges: divide by the number of symmetry operations in the group
    (iter_chg->second) /= double(_cavity->pointGroup().nrIrrep());
}

void compute_polarization_energy(double * energy)
{
    // Check if NucMEP && EleASC surface functions exist.
    bool is_separate = (surfaceFunctionExists(functions, "NucMEP") && surfaceFunctionExists(functions, "EleASC"));

    if (is_separate) {
        // Using separate potentials and charges
        SurfaceFunctionMap::const_iterator iter_nuc_pot = functions.find("NucMEP");
        SurfaceFunctionMap::const_iterator iter_nuc_chg = functions.find("NucASC");
        SurfaceFunctionMap::const_iterator iter_ele_pot = functions.find("EleMEP");
        SurfaceFunctionMap::const_iterator iter_ele_chg = functions.find("EleASC");

        double UNN = (iter_nuc_pot->second) * (iter_nuc_chg->second);
        double UEN = (iter_ele_pot->second) * (iter_nuc_chg->second);
        double UNE = (iter_nuc_pot->second) * (iter_ele_chg->second);
        double UEE = (iter_ele_pot->second) * (iter_ele_chg->second);

        std::ostringstream out_stream;
        out_stream << "Polarization energy components" << std::endl;
        out_stream << "  U_ee = " << boost::format("%20.14f\n") % (UEE / 2.0);
        out_stream << "  U_en = " << boost::format("%20.14f\n") % (UEN / 2.0);
        out_stream << "  U_ne = " << boost::format("%20.14f\n") % (UNE / 2.0);
        out_stream << "  U_nn = " << boost::format("%20.14f\n") % (UNN / 2.0);
        out_stream << "  U_en - U_ne = " << boost::format("%20.14E\n") % (UEN - UNE);
        printer(out_stream);

        *energy = 0.5 * ( UNN + UEN + UNE + UEE );
    } else {
        SurfaceFunctionMap::const_iterator iter_pot = functions.find("TotMEP");
        SurfaceFunctionMap::const_iterator iter_chg = functions.find("TotASC");

        *energy = (iter_pot->second) * (iter_chg->second);
        *energy /= 2.0;
    }
}

void save_surface_functions()
{
    printer("\nDumping surface functions to .npy files");
    for (auto pair : functions) {
        unsigned int dim = static_cast<unsigned int>(pair.second.nPoints());
        const unsigned int shape[] = {dim};
        std::string fname = pair.second.name() + ".npy";
        cnpy::npy_save(fname, pair.second.vector().data(), shape, 1, "w", true);
    }
}

void save_surface_function(const char * name)
{
    typedef SurfaceFunctionMap::const_iterator surfMap_iter;
    std::string functionName(name);
    std::string fname = functionName + ".npy";

    surfMap_iter it = functions.find(functionName);
    unsigned int dim = static_cast<unsigned int>(it->second.nPoints());
    const unsigned int shape[] = {dim};
    cnpy::npy_save(fname, it->second.vector().data(), shape, 1, "w", true);
}

void load_surface_function(const char * name)
{
    std::string functionName(name);
    printer("\nLoading surface function " + functionName + " from .npy file");
    std::string fname = functionName + ".npy";
    cnpy::NpyArray raw_surfFunc = cnpy::npy_load(fname);
    int dim = raw_surfFunc.shape[0];
    if (dim != _cavity->size()) {
        PCMSOLVER_ERROR("Inconsistent dimension of loaded surface function!");
    } else {
        Eigen::VectorXd values = getFromRawBuffer<double>(dim, 1, raw_surfFunc.data);
        SurfaceFunction func(functionName, dim, values);
        // Append to global map
        SurfaceFunctionMap::iterator iter = functions.lower_bound(functionName);
        if ( iter == functions.end()  ||  iter->first != functionName ) {
            if ( iter != functions.begin() ) --iter;
            auto insertion = SurfaceFunctionMap::value_type(functionName, func);
            iter = functions.insert(iter, insertion);
        }
   }
}

void dot_surface_functions(double * result, const char * potString,
                                      const char * chgString)
{
    // Convert C-style strings to std::string
    std::string potFuncName(potString);
    std::string chgFuncName(chgString);

    // Setup iterators
    SurfaceFunctionMap::const_iterator iter_pot = functions.find(potFuncName);
    SurfaceFunctionMap::const_iterator iter_chg = functions.find(chgFuncName);

    if ( iter_pot == functions.end()  ||  iter_chg == functions.end() ) {
        PCMSOLVER_ERROR("One or both of the SurfaceFunction specified is non-existent.");
    } else {
        // Calculate the dot product
        *result = (iter_pot->second) * (iter_chg->second);
    }
}

void get_cavity_size(int * nts, int * ntsirr)
{
    *nts    = _cavity->size();
    *ntsirr = _cavity->irreducible_size();
}

void get_tesserae(double * centers)
{
    // Use some Eigen magic
    for ( int i = 0; i < (3 * _cavity->size()); ++i) {
        centers[i] = *(_cavity->elementCenter().data() + i);
    }
}

void get_tesserae_centers(int * its, double * center)
{
    Eigen::Vector3d tess = _cavity->elementCenter(*its-1);
    center[0] = tess(0);
    center[1] = tess(1);
    center[2] = tess(2);
}

void print_citation()
{
    printer(citation_message());
}

void print_pcm()
{
    std::ostringstream out_stream;
    out_stream << "\n" << std::endl;
    out_stream << "~~~~~~~~~~ PCMSolver ~~~~~~~~~~" << std::endl;
    out_stream << "Using CODATA " << parsedInput->CODATAyear() << " set of constants." << std::endl;
    out_stream << "Input parsing done " << parsedInput->providedBy() << std::endl;
    out_stream << "========== Cavity " << std::endl;
    out_stream << *_cavity << std::endl;
    out_stream << "========== Solver " << std::endl;
    out_stream << *_solver << std::endl;
    out_stream << "============ Medium " << std::endl;
    if (parsedInput->fromSolvent()) {
        out_stream << "Medium initialized from solvent built-in data." << std::endl;
        Solvent solvent = parsedInput->solvent();
        out_stream << solvent << std::endl;
    }
    out_stream << _solver->printGreensFunctions() << std::endl;
    printer(out_stream);
}

void set_surface_function(int * nts, double * values, char * name)
{
    int nTess = _cavity->size();
    if ( nTess != *nts )
        PCMSOLVER_ERROR("You are trying to allocate a SurfaceFunction bigger than the cavity!");

    std::string functionName(name);
    // Here we check whether the function exists already or not
    // 1. find the lower bound of the map
    SurfaceFunctionMap::iterator iter = functions.lower_bound(functionName);
    // 2. if iter == end, or if iter is not a match,
    //    then this element was not in the map, so we need to insert it
    if ( iter == functions.end()  ||  iter->first != functionName ) {
        // move iter to the element preceeding the insertion point
        if ( iter != functions.begin() ) --iter;
        // insert it
        SurfaceFunction func(functionName, *nts, values);
        auto insertion = SurfaceFunctionMap::value_type(functionName, func);
        iter = functions.insert(iter, insertion);
    } else {
        iter->second.setValues(values);
    }
}

void get_surface_function(int * nts, double * values, char * name)
{
    int nTess = _cavity->size();
    if ( nTess != *nts )
        PCMSOLVER_ERROR("You are trying to access a SurfaceFunction bigger than the cavity!");

    std::string functionName(name);

    SurfaceFunctionMap::const_iterator iter = functions.find(functionName);
    if ( iter == functions.end() )
        PCMSOLVER_ERROR("You are trying to access a non-existing SurfaceFunction.");

    for ( int i = 0; i < nTess; ++i ) {
        values[i] = iter->second.value(i);
    }
}

void add_surface_function(char * result, double * coeff, char * part)
{
    std::string resultName(result);
    std::string partName(part);

    append_surface_function(result);

    SurfaceFunctionMap::iterator iter_part = functions.find(partName);
    SurfaceFunctionMap::iterator iter_result = functions.find(resultName);

    (iter_result->second) += (*coeff) * (iter_part->second);
}

void print_surface_function(char * name)
{
    std::string functionName(name);

    SurfaceFunctionMap::iterator iter = functions.find(name);
    std::ostringstream out_stream;
    out_stream << "\n" << std::endl;
    out_stream << iter->second << std::endl;
    printer(out_stream);
}

void clear_surface_function(char * name)
{
    std::string functionName(name);

    SurfaceFunctionMap::iterator iter = functions.find(name);
    // Clear contents if found
    if (iter != functions.end()) iter->second.clear();
}

void append_surface_function(char * name)
{
    int nTess = _cavity->size();
    std::string functionName(name);

    // Here we check whether the function exists already or not
    // 1. find the lower bound of the map
    SurfaceFunctionMap::iterator iter = functions.lower_bound(functionName);
    // 2. if iter == end, or if iter is not a match,
    //    then this element was not in the map, so we need to insert it
    if ( iter == functions.end()  ||  iter->first != functionName ) {
        // move iter to the element preceeding the insertion point
        if ( iter != functions.begin() ) --iter;
        // insert it
        SurfaceFunction func(functionName, nTess);
        auto insertion = SurfaceFunctionMap::value_type(functionName, func);
        iter = functions.insert(iter, insertion);
    } else {
        // What happens if it is already in the map? The values need to be updated.
        // Nothing, I assume that if one calls append_surface_function will then also call
        // set_surface_function somewhere else, hence the update will be done there.
    }
}

void scale_surface_function(char * func, double * coeff)
{
    std::string resultName(func);

    SurfaceFunctionMap::iterator iter_func = functions.find(func);

    // Using iterators and operator overloading: so neat!
    iter_func->second *= (*coeff);
}

void push_input_string(char * s)
{
	std::string str(s);
	// Save the string inside a std::vector<std::string>
	input_strings.push_back(str);
}

#ifdef __cplusplus
}
#endif

/*

	Functions not visible to host program

*/

void setupInput(bool from_host)
{
    if (from_host) { // Set up input from host data structures
        cavityInput cav;
        cav.cleaner();
        solverInput solv;
        solv.cleaner();
        greenInput green;
        green.cleaner();
        host_input(&cav, &solv, &green);
        // Put string passed with the alternative method in the input structures
        if (!input_strings.empty()) {
            for (size_t i = 0; i < input_strings.size(); ++i) {
                // Trim strings aka remove blanks
                boost::algorithm::trim(input_strings[i]);
            }
            strncpy(cav.cavity_type,    input_strings[0].c_str(), input_strings[0].length());
            strncpy(cav.radii_set,      input_strings[1].c_str(), input_strings[1].length());
            strncpy(cav.restart_name,   input_strings[2].c_str(), input_strings[2].length());
            strncpy(solv.solver_type,   input_strings[3].c_str(), input_strings[3].length());
            strncpy(solv.solvent,       input_strings[4].c_str(), input_strings[4].length());
            strncpy(solv.equation_type, input_strings[5].c_str(), input_strings[5].length());
            strncpy(green.inside_type,  input_strings[6].c_str(), input_strings[6].length());
            strncpy(green.outside_type, input_strings[7].c_str(), input_strings[7].length());
        }
        parsedInput = pcm::make_shared<Input>(Input(cav, solv, green));
    } else {
	    parsedInput = pcm::make_shared<Input>(Input("@pcmsolver.inp"));
    }
    std::string _mode = parsedInput->mode();
    // The only thing we can't create immediately is the molecule
    // from which the cavity is to be built.
    if (_mode != "EXPLICIT") {
       Molecule molec;
       initMolecule(molec);
       parsedInput->molecule(molec);
    }
}

void initCavity()
{
    _cavity = Factory<Cavity, cavityData>::TheFactory().create(parsedInput->cavityType(), parsedInput->cavityParams());
}

void initSolver()
{
    // INSIDE
    SharedIGreensFunction gf_i = Factory<IGreensFunction, greenData>::TheFactory().create(parsedInput->greenInsideType(),
		                                              parsedInput->insideGreenParams());
    // OUTSIDE
    SharedIGreensFunction gf_o = Factory<IGreensFunction, greenData>::TheFactory().create(parsedInput->greenOutsideType(),
		                                              parsedInput->outsideStaticGreenParams());
    std::string modelType = parsedInput->solverType();
    _solver = Factory<PCMSolver, solverData, SharedIGreensFunction, SharedIGreensFunction>::TheFactory().create(modelType, parsedInput->solverParams(), gf_i, gf_o);
    _solver->buildSystemMatrix(*_cavity);
    // Always save the cavity in a cavity.npz binary file
    // Cavity should be saved to file in initCavity(), due to the dependencies of
    // the WaveletCavity on the wavelet solvers it has to be done here...
    _cavity->saveCavity();
}

void initNonEqSolver()
{
    // INSIDE
    SharedIGreensFunction gf_i = Factory<IGreensFunction, greenData>::TheFactory().create(parsedInput->greenInsideType(),
		                                              parsedInput->insideGreenParams());
    // OUTSIDE
    SharedIGreensFunction gf_o = Factory<IGreensFunction, greenData>::TheFactory().create(parsedInput->greenOutsideType(),
		                                               parsedInput->outsideDynamicGreenParams());
    std::string modelType = parsedInput->solverType();
    _noneqSolver = Factory<PCMSolver, solverData, SharedIGreensFunction, SharedIGreensFunction>::TheFactory().create(modelType, parsedInput->solverParams(), gf_i, gf_o);
    _noneqSolver->buildSystemMatrix(*_cavity);
    noneqExists = true;
}

void initAtoms(Eigen::VectorXd & charges_, Eigen::Matrix3Xd & sphereCenter_)
{
    int nuclei;
    collect_nctot(&nuclei);
    sphereCenter_.resize(Eigen::NoChange, nuclei);
    charges_.resize(nuclei);
    double * chg = charges_.data();
    double * centers = sphereCenter_.data();
    collect_atoms(chg, centers);
}

void initMolecule(Molecule & molecule_)
{
    // Gather information necessary to build molecule_
    // 1. number of atomic centers
    int nuclei;
    collect_nctot(&nuclei);
    // 2. position and charges of atomic centers
    Eigen::Matrix3Xd centers;
    centers.resize(Eigen::NoChange, nuclei);
    Eigen::VectorXd charges  = Eigen::VectorXd::Zero(nuclei);
    double * chg = charges.data();
    double * pos = centers.data();
    collect_atoms(chg, pos);
    // 3. list of atoms and list of spheres
    bool scaling = parsedInput->scaling();
    std::string set = parsedInput->radiiSet();
    double factor = angstromToBohr(parsedInput->CODATAyear());
    std::vector<Atom> radiiSet, atoms;
    if ( set == "UFF" ) {
        radiiSet = Atom::initUFF();
    } else {
        radiiSet = Atom::initBondi();
    }
    std::vector<Sphere> spheres;
    for (int i = 0; i < charges.size(); ++i) {
        int index = int(charges(i)) - 1;
	atoms.push_back(radiiSet[index]);
        double radius = radiiSet[index].atomRadius() * factor;
        if (scaling) {
            radius *= radiiSet[index].atomRadiusScaling();
        }
        spheres.push_back(Sphere(centers.col(i), radius));
    }
    // 4. masses
    Eigen::VectorXd masses = Eigen::VectorXd::Zero(nuclei);
    for (int i = 0; i < masses.size(); ++i) {
	 masses(i) = atoms[i].atomMass();
    }
    // Based on the creation mode (Implicit or Atoms)
    // the spheres list might need postprocessing
    std::string _mode = parsedInput->mode();
    if ( _mode == "ATOMS" ) {
       initSpheresAtoms(centers, spheres);
    }
    // 5. molecular point group
    int nr_gen;
    int gen1, gen2, gen3;
    set_point_group(&nr_gen, &gen1, &gen2, &gen3);
    Symmetry pg = buildGroup(nr_gen, gen1, gen2, gen3);

    // OK, now get molecule_
    molecule_ = Molecule(nuclei, charges, masses, centers, atoms, spheres, pg);
}

void initSpheresAtoms(const Eigen::Matrix3Xd & sphereCenter_,
                      std::vector<Sphere> & spheres_)
{
    vector<int> atomsInput = parsedInput->atoms();
    vector<double> radiiInput = parsedInput->radii();

    // Loop over the atomsInput array to get which atoms will have a user-given radius
    for (size_t i = 0; i < atomsInput.size(); ++i) {
        int index = atomsInput[i] - 1; // -1 to go from human readable to machine readable
        // Put the new Sphere in place of the implicit-generated one
        spheres_[index] = Sphere(sphereCenter_.col(index), radiiInput[i]);
    }
}

void initSpheresImplicit(const Eigen::VectorXd & charges_,
                         const Eigen::Matrix3Xd & sphereCenter_, std::vector<Sphere> & spheres_)
{
    bool scaling = parsedInput->scaling();
    std::string set = parsedInput->radiiSet();
    double factor = angstromToBohr(parsedInput->CODATAyear());

    std::vector<Atom> radiiSet;
    if ( set == "UFF" ) {
        radiiSet = Atom::initUFF();
    } else {
        radiiSet = Atom::initBondi();
    }

    for (int i = 0; i < charges_.size(); ++i) {
        int index = charges_(i) - 1;
        double radius = radiiSet[index].atomRadius() * factor;
        if (scaling) {
            radius *= radiiSet[index].atomRadiusScaling();
        }
        spheres_.push_back(Sphere(sphereCenter_.col(i), radius));
    }
}

bool surfaceFunctionExists(const SurfaceFunctionMap & sf_map, const std::string & name)
{
    SurfaceFunctionMap::const_iterator iter = sf_map.find(name);

    return (iter != sf_map.end());
}

void insertSurfaceFunction(SurfaceFunctionMap & sf_map, const SurfaceFunction & sf)
{
    sf_map.insert(std::make_pair(sf.name(), sf));
}

inline void printer(const std::string & message)
{
    // Extract C-style string from C++-style string and get its length
    const char * message_C = message.c_str();
    size_t message_length = strlen(message_C);
    // Call the host_writer
    host_writer(message_C, &message_length);
}

inline void printer(std::ostringstream & stream)
{
    // Extract C++-style string from stream
    std::string message = stream.str();
    // Extract C-style string from C++-style string and get its length
    const char * message_C = message.c_str();
    size_t message_length = strlen(message_C);
    // Call the host_writer
    host_writer(message_C, &message_length);
}
