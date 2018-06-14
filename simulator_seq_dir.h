/*
    ABCD-GPU: Simulating Population Dynamics P systems on the GPU, by DCBA 
    ABCD-GPU is a subproject of PMCGPU (Parallel simulators for Membrane 
                                        Computing on the GPU)   
 
    Copyright (c) 2018  Research Group on Natural Computing, Universidad de Sevilla
    					Dpto. Ciencias de la Computación e Inteligencia Artificial
    					Escuela Técnica Superior de Ingeniería Informática,
    					Avda. Reina Mercedes s/n, 41012 Sevilla (Spain)

    Author: Miguel Ángel Martínez-del-Amor
    
    This file is part of ABCD-GPU.
  
    ABCD-GPU is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ABCD-GPU is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with ABCD-GPU.  If not, see <http://www.gnu.org/licenses/>. */

#ifndef __SEQ_SIM__DIR__
#define __SEQ_SIM__DIR__

#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "pdp_psystem_sab.h"
#include "simulator.h"


class Simulator_seq_dir: public Simulator {

	//friend class PDP_Psystem_SAB;

public:

	Simulator_seq_dir(PDP_Psystem_SAB* PDPps) {
		this->structures=PDPps->structures;
		this->options=PDPps->options;
	}

	bool run();
	bool step(int k);
	
	/* Methods for querying data */
	unsigned int get_object_value(  int obj, 
					int membr, 
					int env,
					int sim) {
		return structures->configuration.multiset[sim*options->num_environments*options->num_objects*options->num_membranes+env*options->num_objects*options->num_membranes+membr*options->num_objects+obj];
	}

	char get_membrane_charge(  int membr,
				   int env,
				   int sim) {
		return structures->configuration.membrane[sim*options->num_environments*options->num_membranes+env*options->num_membranes+membr];
	}

private:

	PDP_Psystem_SAB::Structures structures;
	Options options;

	/*****************************/
	/* Auxiliary data structures */
	/*****************************/

	float *addition;
	bool *active_block;
	unsigned int *block_min;
	PDP_Psystem_SAB::Configuration ini_cfg;
	unsigned int msize,esize,bpsize,besize,rpsize,resize;

	/* Only debugging purposes */
	void print_configuration();
	void print_activations();
	void print_block_applications();

	/* Initialization */
	void init();
	void reset();
	void del();

	/* Micro phases */
	void selection_phase1(unsigned int simulation, unsigned int environment);
	void selection_phase2(unsigned int simulation, unsigned int environment);
	void selection_phase3(unsigned int simulation, unsigned int environment);

	/* Main phases */
	unsigned int selection(unsigned int simulation, unsigned int environment);
	unsigned int execution(unsigned int simulation, unsigned int environment);
	unsigned int execution_communication(unsigned int sim);
};


/* Simulation algorithm based on a distribution table */
//bool simulation_seq_table(Structures* structures,Options* options);

/* Simulation algorithm based on a direct distribution */
//bool simulation_seq_dir(Structures* structures, Options* options);

#endif
