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
void print_comparing_partition(int* partition, int* alphabet,int* partition2, int* alphabet2);
void print_rules(unsigned int* rules_size,unsigned int* lhs_object,int num_rules,int num_objects);
void reset_partition(int* partition,int* alphabet,int num_rules,int num_objects);
int initialize_rules(int *data, int size);
void initialize_lhs(int *data, int size);
void make_partition(int* partition,unsigned int* rules_size,unsigned int*lhs_object, int total_lhs,int * alphabet,int num_rules,int num_objects);

void make_partition(int* partition,unsigned int* rules_size,unsigned int*lhs_object, int total_lhs,int * alphabet,int num_rules,int num_objects,unsigned int *membrane, unsigned int* mmultiplicity);
void make_partition_2(int* partition, int* rules_size, int*lhs_object, int total_lhs,int * alphabet);
void make_partition_gpu(int* partition, int* rules_size, int*lhs_object, int total_lhs,int * alphabet,bool version2=false);

bool check_compete(int block_a,int block_b,unsigned int* rules_size,unsigned int * lhs_object,unsigned int *membrane, unsigned int* mmultiplicity);
bool check_compete(int block_a,int block_b,unsigned int* rules_size,unsigned int * lhs_object);


void compare_partition(int* partition, int* alphabet,int* partition2, int* alphabet2,int num_rules,int num_objects);
int normalize_partition(int* partition, int* trans_partition,int size);
}

#endif /* COMPETITION_H_ */
