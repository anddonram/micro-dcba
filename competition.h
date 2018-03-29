/*
 * competition.h
 *
 *  Created on: 7/3/2018
 *      Author: andres
 */

#ifndef COMPETITION_H_
#define COMPETITION_H_
namespace competition{

void print_header();
void print_partition(int* partition, int* alphabet,int num_rules,int num_objects);
void print_comparing_partition(int* partition, int* alphabet,int* partition2, int* alphabet2,int num_rules,int num_objects);
void print_rules(unsigned int* rules_size,unsigned int* lhs_object,int num_rules,int num_objects);
void reset_partition(int* partition,int* alphabet,int num_rules,int num_objects);
int initialize_rules(int *data, int size);
void initialize_lhs(int *data, int size);

//Old version
void make_partition(int* partition,unsigned int* rules_size,unsigned int*lhs_object,int * alphabet,int num_rules,int num_objects);
//New version, takes membranes into account
void make_partition(int* partition,unsigned int* rules_size,unsigned int*lhs_object,int * alphabet,int num_rules,int num_objects,unsigned int *membrane, unsigned int* mmultiplicity);

void make_partition_gpu(int* partition,unsigned int* rules_size,unsigned int*lhs_object,int * alphabet,int num_rules,int num_objects,int num_membranes ,unsigned int* mmultiplicity,int mult_size);

/*Identical version to the GPU, except sequential
 *
 */
void make_partition_2(int* partition,
		unsigned int* rules_size,
		unsigned int* lhs_object,
		int * alphabet,
		int num_rules,
		int num_objects,
		int num_membranes,
		unsigned int* mmultiplicity,
		int mult_size);

//New version, takes membranes into account
bool check_compete(int block_a,int block_b,unsigned int* rules_size,unsigned int * lhs_object,unsigned int *membrane, unsigned int* mmultiplicity);
//Old version
bool check_compete(int block_a,int block_b,unsigned int* rules_size,unsigned int * lhs_object);


void compare_partition(int* partition, int* alphabet,int* partition2, int* alphabet2,int num_rules,int num_objects);
int normalize_partition(int* partition, int* trans_partition,int size);
int initialize_partition_structures(int* partition,
		int num_partitions,
		int num_rules,
		int** accum_offsets,
		int** ordered);
}

#endif /* COMPETITION_H_ */
