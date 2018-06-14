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

#include "simulator_omp_dir.h"

#include <math.h>
#include <limits.h>
#include <iostream>
#include <timestat.h>

// TODO: Unify this
#define IS_PI_BLOCK_ACTIVE(b) ((b)>=0 && (b)<bpsize && active_block[(sim*env_parallel*besize+etid*besize+(b))])
#define IS_ENV_BLOCK_ACTIVE(b) ((b)<besize && (b)>=bpsize && active_block[(sim*env_parallel*besize+etid*besize+(b))])

/* The following sets how the arrays are indexed */
#define AB_IDX (sim*env_parallel*besize+etid*besize+block)
#define BM_IDX (sim*env_parallel*besize+etid*besize+block)
#define AD_IDX(o,m) (sim*env_parallel*esize+etid*esize+(m)*msize+(o))
//#define NB_P_IDX ((mode==0)?(env*bpsize+block):(sim*(options->num_environments*bpsize+(besize-bpsize))+env*bpsize+block))
//#define NB_E_IDX ((mode==0)?(options->num_environments*bpsize+(block-bpsize)):(sim*(options->num_environments*bpsize+(besize-bpsize))+options->num_environments*bpsize+(block-bpsize)))
#define NB_IDX (sim*env_parallel*besize+etid*besize+block)
//#define NB_E_IDX ((mode==0)?(tid*besize+block):(sim*(env_parallel*besize)+block))
#define NR_P_IDX (sim*(options->num_environments*rpsize+(resize-rpsize))+env*rpsize+r)
#define NR_E_IDX (sim*(options->num_environments*rpsize+(resize-rpsize))+options->num_environments*rpsize+(r-rpsize))
#define CH_IDX(m) (sim*(options->num_environments*options->num_membranes)+env*options->num_membranes+(m))
#define MU_IDX(o,m) (sim*options->num_environments*esize+env*esize+(m)*msize+(o))


using namespace std;

/*************/
/* MAIN LOOP */
/*************/

bool Simulator_omp_dir::run() {
	if (mode==0)
		return run_parallel_env();
	else if (mode==1)
		return run_parallel_sim();
	else if (mode==2||mode==3)
		return run_parallel_hybrid();
	else if (mode==4)
			return run_parallel_env_deprecated();
	return false;
}

bool Simulator_omp_dir::run_parallel_env(){
	mode=0;

	init();

	if (options->verbose>0)
		cout << "[2] STARTING THE SIMULATION: Using virtual table (direct) algorithm, with OpenMP and parallel environments, old" << endl;

	//init_time();
	start_timer();
	/* Repeat for each simulation */
	for (int simu=0; simu<options->num_simulations; simu++) {

		reset();

		/* MAIN LOOP */
		for (int i=0; i<options->time; i++) {

			/* Selection for each environment */
			int env;
			#pragma omp parallel private(env)
			{
			#pragma omp for schedule(dynamic,1)
			for (env=0; env<options->num_environments; env++) {
				selection(0,env);
			}
			#pragma omp barrier
			/* Execution for each environment */
			#pragma omp for schedule(dynamic,1)
			for (env=0; env<options->num_environments; env++) {
				execution(0,env);
			}
			}
			execution_communication(0);

			if (options->verbose>0) cout << "Simulation " << simu << ", Step " << i << endl;
			if (options->verbose>1) print_configuration();
		}
	}
	//print_time(stdout);
	double d=end_timer();
	cout << d << " " << options->mem << endl;

	del();

	return true;
}

// TODO: check whether it is bad or not
bool Simulator_omp_dir::run_parallel_env_deprecated(){
	mode=0;

	init();

	if (options->verbose>0)
		cout << "[2] STARTING THE SIMULATION: Using virtual table (direct) algorithm, with OpenMP and parallel environments, new" << endl;

	//int start=omp_get_wtime();
	start_timer();//init_time();

	int env,simu,i;
	/* Repeat for each simulation */
	#pragma omp parallel private(env,simu,i)
	{
	for (simu=0; simu<options->num_simulations; simu++) {

		reset();

		/* MAIN LOOP */
		for (i=0; i<options->time; i++) {

			/* Selection for each environment */
			#pragma omp for schedule(dynamic,1)
			for (env=0; env<options->num_environments; env++) {
				selection(0,env);
			}

			/* Execution for each environment */
			#pragma omp for schedule(dynamic,1)
			for (env=0; env<options->num_environments; env++) {
				execution(0,env);
			}

			execution_communication(0);

			#pragma omp single
			if (options->verbose>0) cout << "Simulation " << simu << ", Step " << i << endl;
			if (options->verbose>1) print_configuration();
		}
	}
	}
	double d=end_timer();//print_time(stdout);

	cout << d << " " << options->mem << endl;

	del();

	return true;
}

bool Simulator_omp_dir::run_parallel_sim(){

	init();

	if (options->verbose>0)
		cout << "[2] STARTING THE SIMULATION: Using virtual table (direct) algorithm, with OpenMP and parallel simulations" << endl;

	/* Repeat for each simulation */
	start_timer();//init_time();
	#pragma omp parallel for schedule (dynamic,1)
	for (int simu=0; simu<options->num_simulations; simu++) {
		int stid=omp_get_thread_num();
		reset(stid);

		/* MAIN LOOP */
		for (int i=0; i<options->time; i++) {

			/* Selection for each environment */
			for (int env=0; env<options->num_environments; env++) {
				selection(stid,env);
			}

			/* Execution for each environment */
			for (int env=0; env<options->num_environments; env++) {
				execution(stid,env);
			}

			execution_communication(stid);

			if (options->verbose>0) cout << "Simulation " << simu << ", Step " << i << endl;
			if (options->verbose>1) print_configuration();
		}
	}

	double d=end_timer();//print_time(stdout);
	cout << d << " " << options->mem << endl;

	del();

	return true;
}

// TODO: check whether it is bad or not
bool Simulator_omp_dir::run_parallel_hybrid(){
//	mode=1;

	init();

	omp_set_nested(1);

	if (options->verbose>0)
		cout << "[2] STARTING THE SIMULATION: Using virtual table (direct) algorithm, with OpenMP and hybrid parallelism" << endl;

	start_timer();//init_time();

	/* Repeat for each simulation */
	#pragma omp parallel for schedule (dynamic,1) num_threads (sim_parallel)
	for (int simu=0; simu<options->num_simulations; simu++) {

		int stid=omp_get_thread_num();

		#pragma omp parallel num_threads(env_parallel)
		{
		reset(stid);

		/* MAIN LOOP */
		for (int i=0; i<options->time; i++) {

			/* Selection for each environment */
			#pragma omp for schedule(dynamic,1)
			for (int env=0; env<options->num_environments; env++) {
				selection(stid,env);
			}

			/* Execution for each environment */
			#pragma omp for schedule(dynamic,1)
			for (int env=0; env<options->num_environments; env++) {
				execution(stid,env);
			}

			execution_communication(stid);

			#pragma omp single
			if (options->verbose>0) cout << "Simulation " << simu << ", Step " << i << endl;
			if (options->verbose>1) print_configuration();
		}
		}
	}

	double d=end_timer();//print_time(stdout);

	cout << d << " " << options->mem << endl;

	del();

	return true;
}

/***************************************/
/* Auxiliary functions Implementations */
/***************************************/

void Simulator_omp_dir::init() {

	/* Configuring threads */
	//omp_set_nested(1);

	max_threads = omp_get_max_threads();

	if (mode==0) {
		env_parallel=gsl_min(max_threads,options->num_environments);
		sim_parallel=1;
	}
	else if (mode==1) {
		env_parallel=1;
		sim_parallel=gsl_min(max_threads,options->num_simulations);
	}
	else if (mode==2) {
		max_threads=gsl_min(max_threads,options->num_simulations);
		sim_parallel=(max_threads>1)?2:1;
		env_parallel=max_threads/sim_parallel;
	}
	else if (mode==3) {
		max_threads=gsl_min(max_threads,options->num_environments);
		env_parallel=(max_threads>1)?2:1;
		sim_parallel=max_threads/env_parallel;
	}
	else {
		env_parallel=1;
		sim_parallel=1;
	}

	/* Create initial configuration */
	ini_cfg = structures->configuration;

	structures->configuration.membrane_size=options->num_membranes*options->num_environments*sim_parallel;
	structures->configuration.membrane=new PDP_Psystem_SAB::CHARGE[structures->configuration.membrane_size];

	structures->configuration.multiset_size = options->num_objects*options->num_membranes*options->num_environments*sim_parallel;
	structures->configuration.multiset = new PDP_Psystem_SAB::MULTIPLICITY[structures->configuration.multiset_size];

	/* Initialize Nb */
	structures->nb_size= (options->num_rule_blocks+options->num_blocks_env)*env_parallel*sim_parallel;
	structures->nb = new PDP_Psystem_SAB::MULTIPLICITY [structures->nb_size];

	/* Initialize Nr */
	structures->nr_size= ((structures->rule_size)*options->num_environments+structures->env_rule_size)*sim_parallel;
	structures->nr = new PDP_Psystem_SAB::MULTIPLICITY [structures->nr_size];

	/* Create and initialize auxiliary structures */

	unsigned int addition_size=options->num_objects*options->num_membranes*env_parallel*sim_parallel;
	/* Initialize auxiliary structures */
	addition = new float[addition_size];
	for (unsigned int i=0; i<addition_size; i++)
		addition[i]=0.0;

	unsigned int active_block_size=(options->num_rule_blocks+options->num_blocks_env)*env_parallel*sim_parallel;
	active_block = new bool[active_block_size];
	for (unsigned int i=0; i<active_block_size; i++)
		active_block[i]=false;

	unsigned int block_min_size=(options->num_rule_blocks+options->num_blocks_env)*env_parallel*sim_parallel;
	block_min = new unsigned int[block_min_size];
	for (unsigned int i=0; i<block_min_size; i++)
		block_min[i]=0;

	/* Set auxiliary sizes info */
	esize=options->num_objects*options->num_membranes;
	msize=options->num_objects;
	bpsize=options->num_rule_blocks;
	besize=options->num_blocks_env+options->num_rule_blocks;
	rpsize=structures->rule_size;
	resize=structures->rule_size+structures->env_rule_size;

	unsigned long int mem=addition_size*sizeof(float)+active_block_size*sizeof(bool)+block_min_size*sizeof(int)+structures->nb_size*sizeof(unsigned int)+
					structures->nr_size*sizeof(unsigned int)+structures->configuration.membrane_size*sizeof(char)+structures->configuration.multiset_size*sizeof(unsigned int);
	options->mem+=mem;

	/* Print new information */
	if (options->verbose>0) {
		cout << endl << "--------------------------" << endl << "Memory used by the virtual table algorithm:" << endl;

		cout << "Addition: " << addition_size*sizeof(float) << " (" << addition_size*sizeof(float)/1024 << "KB)" << endl;
		cout << "Active blocks: " << active_block_size*sizeof(bool) << " (" << active_block_size*sizeof(bool)/1024 << "KB)" << endl;
		cout << "Block min: " << block_min_size*sizeof(int) << " (" << block_min_size*sizeof(int)/1024 << "KB)" << endl;
		cout << "Nb: " << structures->nb_size*sizeof(unsigned int) << " (" << structures->nb_size*sizeof(unsigned int)/1024 << "KB)" << endl;
		cout << "Nr: " << structures->nr_size*sizeof(unsigned int) << " (" << structures->nr_size*sizeof(unsigned int)/1024 << "KB)" << endl;
		cout << "Membrane charges: " << structures->configuration.membrane_size*sizeof(char) << " (" << structures->configuration.membrane_size*sizeof(char)/1024 << "KB)" << endl;
		cout << "Multisets: " << structures->configuration.multiset_size*sizeof(unsigned int) << " (" << structures->configuration.multiset_size*sizeof(unsigned int)/1024 << "KB)" << endl;

		int count=0;
		float div=1;
		char unit[6]={' ','K','M','G','T','P'};
		while ((mem/div)>1023 && count<3){
			div*=1024;
			count++;
		}
		cout << "TOTAL: " << mem << " (" << mem/div << " " << unit[count] << "B)" << endl << endl;
	}
}

void Simulator_omp_dir::del() {
	delete [] addition;
	delete [] active_block;
	delete [] block_min;
	PDP_Psystem_SAB::Configuration aux;
	aux=structures->configuration;
	structures->configuration=ini_cfg;
	delete []aux.membrane;
	delete []aux.multiset;
	delete []structures->nb;
	delete []structures->nr;
}

void Simulator_omp_dir::reset(unsigned int sim) {

	if (mode==0) {
		/* Re-init configurations */
		#pragma omp for schedule(dynamic,1)
		for (int env=0; env<options->num_environments; env++) {
			for (int m=0;m<options->num_membranes; m++) {
				structures->configuration.membrane[CH_IDX(m)]=ini_cfg.membrane[CH_IDX(m)];
			}

			for (int o=0;o<options->num_membranes*options->num_objects;o++) {
				structures->configuration.multiset[MU_IDX(o,0)]=ini_cfg.multiset[MU_IDX(o,0)];
			}
		}
	}
	else if (mode==1) {
		/* Re-init configurations */
		for (int m=0;m<ini_cfg.membrane_size; m++) {
			structures->configuration.membrane[sim*ini_cfg.membrane_size+m]=ini_cfg.membrane[m];
		}

		for (int o=0;o<ini_cfg.multiset_size;o++) {
			structures->configuration.multiset[sim*ini_cfg.multiset_size+o]=ini_cfg.multiset[o];
		}
	}
	else if (mode==2||mode==3) {
		/* Re-init configurations */
		#pragma omp for schedule(dynamic,1)
		for (int env=0; env<options->num_environments; env++) {
			for (int m=0;m<options->num_membranes; m++) {
				structures->configuration.membrane[CH_IDX(m)]=ini_cfg.membrane[env*options->num_membranes+m];
			}

			for (int o=0;o<options->num_membranes*options->num_objects;o++) {
				structures->configuration.multiset[MU_IDX(o,0)]=ini_cfg.multiset[env*esize+o];
			}
		}
	}
}

/***************/
/* MAIN PHASES */
/***************/

unsigned int Simulator_omp_dir::selection(unsigned int sim, unsigned int env){

	//cout << "Maximum of " << omp_get_max_threads() << " threads" << endl;
	//cout << "Using " << omp_get_num_threads() << " threads" << endl;

	/* PHASE 1: DISTRIBUTION */
	selection_phase1(sim,env);

	/* PHASE 2: MAXIMALITY */
	selection_phase2(sim,env);

	/* PHASE 3: PROBABILITY */
	selection_phase3(sim,env);

	return 1;
}

unsigned int Simulator_omp_dir::execution(unsigned int sim, unsigned int env) {
	/* PHASE 4: EXECUTION */
	/* For each environemnt */

	if (options->verbose>1)
		cout << endl <<endl << "*************************************" << endl << "Phase 4." << endl;

	if (options->verbose>1)
		cout << endl << "Environment " << env << endl;

	for (unsigned int r=0; r<rpsize; r++) {
		/* If there is applications */
		if (structures->nr[NR_P_IDX]>0) {
			if (options->verbose>1) cout << "Rule " << r << endl;
			for (int o=structures->rule.rhs_ini[r]; o<structures->rule.rhs_ini[r]+structures->rule.rhs_length[r]; o++) {
				unsigned int obj=structures->rhs.object[o];
				unsigned int mult=structures->rhs.multiplicity[o];
				unsigned int membr=structures->rhs.membrane[o];

				structures->configuration.multiset[MU_IDX(obj,membr)]+=structures->nr[NR_P_IDX]*mult;

				if (options->verbose>1) cout << "\t adds [obj_" << obj << "]^" << membr << "*" << structures->nr[NR_P_IDX]*mult << endl;
				structures->nr[NR_P_IDX]=0;
			}
		}
	}

	return 0;
}

unsigned int Simulator_omp_dir::execution_communication(unsigned int sim) {

	if (options->verbose>1) cout << endl << "Communication rules" << endl;

	for (unsigned int r=rpsize;r<resize;r++) {
		if (options->verbose>1) cout << "Rule " << r << endl;
		if (structures->nr[NR_E_IDX]>0) {
			for (unsigned int o=structures->rule.rhs_ini[r]; o<structures->rule.rhs_ini[r]+structures->rule.rhs_length[r]; o++) {
				unsigned int obj=structures->rhs.object[o];
				//unsigned int mult=structures->rhs.multiplicity[o];
				unsigned int env=structures->rhs.membrane[o];

				structures->configuration.multiset[MU_IDX(obj,0)]+=structures->nr[NR_E_IDX];
				if (options->verbose>1) cout << "\t adds (obj_" << obj << ")^" << env << "*" << structures->nr[NR_E_IDX] << endl;
				structures->nr[NR_E_IDX]=0;
			}
		}
	}

	return 0;
}


/*********************/
/* Selection methods */
/*********************/

void Simulator_omp_dir::selection_phase1(unsigned int sim, unsigned int env) {
	unsigned int etid=(mode==1)?0:omp_get_thread_num();

	/** Filters 1 and 2 for blocks in Pi **/
	for (unsigned int block=0; block<bpsize; block++) {
		unsigned int am=structures->ruleblock.active_membrane[block];
		char ch=structures->ruleblock.lhs_charge[block];
		/* Re-init */
		active_block[AB_IDX]=true;
		block_min[BM_IDX]=0;
		structures->nb[NB_IDX]=0;

		/* Filter 1: If not equal charges in lhs */
		if (structures->configuration.membrane[CH_IDX(am)] != ch)
			active_block[AB_IDX]=false;
		else {
			/* Filter 2: objects in lhs equals 0 */
			for (unsigned int o=structures->ruleblock.lhs_ini[block]; o<structures->ruleblock.lhs_ini[block]+structures->ruleblock.lhs_length[block]; o++) {
				unsigned int obj=structures->lhs.object[o];
				unsigned int mult=structures->lhs.multiplicity[o];
				unsigned int membr=structures->lhs.membrane[o];

				if (structures->configuration.multiset[MU_IDX(obj,membr)]<mult) {
					active_block[AB_IDX]=false;
					break;
				}
			}
		}
	}

	/** Filters 1 and 2 for communication blocks **/
	for (unsigned int block=bpsize; block<besize; block++) {
		unsigned int am=structures->ruleblock.active_membrane[block];
		//char ch=structures->ruleblock.lhs_charge[block];
		/* Re-init */
		block_min[BM_IDX]=0;
		active_block[AB_IDX]=false;

		/* Filter 1 for environment rules */
		if (am==env) {
			/* Re-init*/
			structures->nb[NB_IDX]=0;

			if (structures->configuration.multiset[MU_IDX(structures->ruleblock.lhs_ini[block],0)]>0) {
				active_block[AB_IDX]=true;
			}
		}
	}

	/* TODO: Generate and select table with mutually consistent blocks */
	/* TODO: Well, at least it'd be interesting to check the mutually consistency */

	int A=options->accuracy;
	bool block_sel=false;

	do {
		/** Normalization - step 1 **/
		for (unsigned int i=0; i<esize; i++) {
			addition[AD_IDX(i,0)]=0.0;
		}

		for (unsigned int block=0; block<besize; block++) {
			// If block for Pi
			if (IS_PI_BLOCK_ACTIVE(block)) {
				for (unsigned int o=structures->ruleblock.lhs_ini[block]; o<structures->ruleblock.lhs_ini[block]+structures->ruleblock.lhs_length[block]; o++) {
					unsigned int obj=structures->lhs.object[o];
					unsigned int mult=structures->lhs.multiplicity[o];
					unsigned int membr=structures->lhs.membrane[o];

					addition[AD_IDX(obj,membr)]+=1.0/(float)mult;
				}
			}
			// If block for communication
			else if (IS_ENV_BLOCK_ACTIVE(block)) {
				addition[AD_IDX(structures->ruleblock.lhs_ini[block],0)]+=1.0;
			}
		}

		/* Normalization - step 2 *
		 * Column minimum calculation */
		for (unsigned int block=0; block<besize; block++) {
			// If block activated, for both Pi and Communication blocks
			if (IS_PI_BLOCK_ACTIVE(block) || IS_ENV_BLOCK_ACTIVE(block)) {
				unsigned int minimum=0;
				bool min_set=false;

				for (unsigned int o=structures->ruleblock.lhs_ini[block]; o<structures->ruleblock.lhs_ini[block]+structures->ruleblock.lhs_length[block]; o++) {
					unsigned int obj=0, mult=0, membr=0;

					if (block>=bpsize) {
						obj=o; mult=1; membr=0;
					}
					else {
						obj=structures->lhs.object[o];
						mult=structures->lhs.multiplicity[o];
						membr=structures->lhs.membrane[o];
					}

					unsigned int value =
							(unsigned int) floor(((double) (1/(double)mult)*(1/(double)mult)
												  / addition[AD_IDX(obj,membr)])
												  * structures->configuration.multiset[MU_IDX(obj,membr)]);
					if (! min_set){
						minimum=value;
						min_set=true;
					}
					if (value<minimum) {
						minimum=value;
					}
				}

				block_min[BM_IDX]=minimum;
			}
		}

		if (options->verbose>1) cout << endl <<endl << "*************************************" << endl << "Phase 1. Environment " << env << ", A=" << A << endl;
#pragma omp critical (PRINT_P1)
		print_activations();

		/* Deleting LHS *
		 * Adding block applications */
		block_sel=false;

		for (unsigned int block=0; block<besize; block++) {
			if (block_min[BM_IDX]>0) {
				block_sel=true;
				/* Consume LHS */
				for (unsigned int o=structures->ruleblock.lhs_ini[block]; o<structures->ruleblock.lhs_ini[block]+structures->ruleblock.lhs_length[block]; o++) {
					unsigned int obj=0, mult=0, membr=0;

					if (block>=bpsize) {
						obj=o; mult=1; membr=0;
					}
					else {
						obj=structures->lhs.object[o];
						mult=structures->lhs.multiplicity[o];
						membr=structures->lhs.membrane[o];
					}

					/* Delete block application */
					structures->configuration.multiset[MU_IDX(obj,membr)]-=block_min[BM_IDX]*mult;
					/* Pre-filter: only filter last rules consuming objects */
					if (structures->configuration.multiset[MU_IDX(obj,membr)]==0)
						active_block[AB_IDX]=false;
				}

				/* Add applications to block */
				structures->nb[NB_IDX]+=block_min[BM_IDX];

				block_min[BM_IDX]=0;
			}
		}

		// TODO: Check how efficient is to keep this filter here, compared with pre-filter
		/* Filter 2: objects in lhs equals 0 */
		for (unsigned int block=0; block<besize; block++) {
			if (IS_PI_BLOCK_ACTIVE(block)||IS_ENV_BLOCK_ACTIVE(block)) {
				for (unsigned int o=structures->ruleblock.lhs_ini[block]; o<structures->ruleblock.lhs_ini[block]+structures->ruleblock.lhs_length[block]; o++) {
					unsigned int obj=0, membr=0;

					if (block>=bpsize) {
						obj=o; membr=0;
					}
					else {
						obj=structures->lhs.object[o];
						membr=structures->lhs.membrane[o];
					}

					if (structures->configuration.multiset[MU_IDX(obj,membr)]==0) {
						active_block[AB_IDX]=false;
						break;
					}
				}
			}
		}

		A--;

	} while (A>0 && block_sel);
}

void Simulator_omp_dir::selection_phase2(unsigned int sim, unsigned int env){
	unsigned int etid=(mode==1)?0:omp_get_thread_num();

	if (options->verbose>1) {
		cout << endl <<endl << "*************************************" << endl << "Phase 2." << endl;
#pragma omp critical (PRINT_P2)
		print_configuration();
	}

	/* For each active block *
	 * In a random order */
	unsigned int nblocks=0; /* Number of active blocks */
	/* Initialize array for random loop */
	unsigned int block_min_idx_aux=sim*env_parallel*besize+etid*besize;
	for (unsigned int block=0; block<besize; block++) {
		if (IS_PI_BLOCK_ACTIVE(block) || IS_ENV_BLOCK_ACTIVE(block)) {
			block_min[block_min_idx_aux+(nblocks++)]=block;
		}
	}

	while (nblocks>0) {
		/* Choose random block */
		unsigned int i=random()%(nblocks--);//--nblocks;
		unsigned int block=block_min[block_min_idx_aux+i];

		/* Pivot it */
		block_min[block_min_idx_aux+i]=block_min[block_min_idx_aux+nblocks];
		block_min[block_min_idx_aux+nblocks]=0;

		unsigned int app=0;
		bool min_set=false;
		/* Check applications (minimum of available objects) */
		for (unsigned int o=structures->ruleblock.lhs_ini[block]; o<structures->ruleblock.lhs_ini[block]+structures->ruleblock.lhs_length[block]; o++) {
			unsigned int obj=0, mult=0, membr=0;

			if (block>=bpsize) {
				obj=o; mult=1; membr=0;
			}
			else {
				obj=structures->lhs.object[o];
				mult=structures->lhs.multiplicity[o];
				membr=structures->lhs.membrane[o];
			}

			unsigned int value=(unsigned int) floor(structures->configuration.multiset[MU_IDX(obj,membr)]/mult);

			if (! min_set){
				app=value;
				min_set=true;
			}
			if (value<app) {
				app=value;
			}
		}

		/* If there are applications */
		if (app>0) {
			structures->nb[NB_IDX]+=app;

			for (unsigned int o=structures->ruleblock.lhs_ini[block]; o<structures->ruleblock.lhs_ini[block]+structures->ruleblock.lhs_length[block]; o++) {
				unsigned int obj=0, mult=0, membr=0;

				if (block>=bpsize) {
					obj=o; mult=1; membr=0;
				}
				else {
					obj=structures->lhs.object[o];
					mult=structures->lhs.multiplicity[o];
					membr=structures->lhs.membrane[o];
				}

				structures->configuration.multiset[MU_IDX(obj,membr)]-=app*mult;
			}
		}
	}

#pragma omp critical (PRINT_P22)
	print_block_applications();
	if (options->verbose>1)  {
#pragma omp critical (PRINT_P23)
		print_configuration();
		cout << endl << endl;
	}

}

void Simulator_omp_dir::selection_phase3(unsigned int sim, unsigned int env) {
	unsigned int etid=(mode==1)?0:omp_get_thread_num();

	if (options->verbose>1)
		cout << endl <<endl << "*************************************" << endl << "Phase 3." << endl;

	/* create a generator chosen by the
	 * environment variable GSL_RNG_TYPE */
	const gsl_rng_type * T_random;
	gsl_rng_env_setup();
	T_random = gsl_rng_default;
	gsl_rng * r_variable = gsl_rng_alloc(T_random);

	if (options->verbose>1) cout << endl << "Environment " << env << endl;

	for (unsigned int block=0; block<besize; block++) {
		int rule_ini=structures->ruleblock.rule_ini[block];
		int rule_length=structures->ruleblock.rule_length[block];

		unsigned int N=0;

		if (block<bpsize) N=structures->nb[NB_IDX];
		else if (env==structures->ruleblock.active_membrane[block]) N=structures->nb[NB_IDX];

		if (options->verbose>1)
			cout << endl << "\tBlock " << block << ", N=" << N << ":";

		if (N==0) {
			for (unsigned int r = rule_ini; r < rule_length+rule_ini; r++) {
				if (r<rpsize) structures->nr[NR_P_IDX] = 0;
				else structures->nr[NR_E_IDX] = 0;

			}
			continue;
		}

		// TODO: check if this is more efficient than adding to a previous phase, or wherever
		/* Execution phase: update charges */
		/*   So execution phase only works with rules information */
		structures->configuration.membrane[structures->ruleblock.active_membrane[block]]=structures->ruleblock.rhs_charge[block];

		float cr=0.0,d=1.0;

		for (unsigned int r = rule_ini; r < rule_length+rule_ini; r++) {
			float p=0.0;
			unsigned int val=0;

			if (r>=rpsize) {
				p=structures->probability[options->num_environments*rpsize+(r-rpsize)];
			}
			else {
				p=structures->probability[env*rpsize+r];
			}

			cr = p / d;

			if (options->verbose>1)
				cout << "r_" << r-rule_ini << "(p=" << p << ",n=";

			if (cr > 0.0) {
				if (r == rule_length+rule_ini-1)
					//structures->nr[env*structures->rule_size+r] = N;
					val=N;
				else {
					val=gsl_ran_binomial (r_variable, (double) cr, N);
					//structures->nr[env*structures->rule_size+r] = gsl_ran_binomial (r_variable, (double) cr, N);
					//cout << "Binomial " << N << ", " << (double) cr << " = " << structures->nr[env*structures->rule_size+r] << endl;
				}
			}

			if (r<rpsize) structures->nr[NR_P_IDX] = val;
			else structures->nr[NR_E_IDX] = val;

			N-=val;
			d*=(1-cr);

			if (options->verbose>1)
				cout << val << "), ";
		}
	}

	gsl_rng_free (r_variable);

}


/***************************/
/* Only debugging purposes */
/***************************/

void Simulator_omp_dir::print_configuration () {

	if (options->verbose==0) return;

	cout << endl << "--------------------------" << endl << "CONFIGURATION" << endl << endl;

	cout << "Multiset" << endl;

	for (int sim=0; sim<options->num_parallel_simulations; sim++) {
		for (int env=0; env<options->num_environments; env++) {
			cout << "Simulation " << sim << ", environment " << env << ":";

			for (int memb=0; memb<options->num_membranes; memb++) {
				cout << endl << "\t\tMembrane " << memb << ": ";
				for (int obj=0; obj<options->num_objects; obj++) {
					cout << "Obj " << obj
							<< "*" << structures->configuration.multiset[sim*options->num_objects*options->num_membranes*options->num_environments+env*options->num_objects*options->num_membranes+memb*options->num_objects+obj] << ", ";
				}
			}

			cout << endl;
		}
	}
	cout << "--------------------------" << endl;
}

void Simulator_omp_dir::print_activations () {
	//unsigned int bpsize=options->num_rule_blocks;
	unsigned int besize=options->num_blocks_env+options->num_rule_blocks;
	unsigned int etid=omp_get_thread_num();

	if (options->verbose<=1) return;
	print_configuration();

	cout << endl << endl << "Activated blocks: ";
	for (int i=0;i<options->num_rule_blocks*options->num_environments+options->num_blocks_env;i++)
		cout << active_block[etid*besize+i] << ", ";

	cout << endl << "Addition vector: ";
	for (int i=0;i<options->num_objects*options->num_membranes;i++)
			cout << addition[etid*besize+i] << ", ";

	cout << endl << "Minimum vector: ";
	for (int i=0;i<options->num_rule_blocks+options->num_blocks_env;i++)
			cout << block_min[etid*besize+i] << ", ";

	cout<<endl;

//print_configuration();
}

void Simulator_omp_dir::print_block_applications() {

	if (options->verbose<=1) return;

	cout << endl << "--------------------------" << endl << "BLOCKS APPLICATIONS" << endl;

	for (unsigned int env=0; env<options->num_environments; env++) {
		cout << endl << "Environment " << env << endl << "\tBlocks: ";
		for (unsigned int block=0; block<options->num_rule_blocks; block++) {
			cout << block << "*" << structures->nb[env*options->num_rule_blocks+block] << ", ";
		}
		for (unsigned int block=options->num_rule_blocks; block<options->num_rule_blocks+options->num_blocks_env; block++) {
			if (structures->ruleblock.active_membrane[block]==env)
				cout << block << "*" << structures->nb[options->num_environments*options->num_rule_blocks+(block-options->num_rule_blocks)] << ", ";
			else cout << block << "*0, ";
		}
	}
	cout << endl;
}
