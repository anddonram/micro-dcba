/*
 ============================================================================
 Name        : competition.cu
 Author      : AndresDR
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include "pdp_psystem_redix.h"
#include "competition.h"
#include <stdio.h>


namespace competition{

__global__ void  make_partition_phase_0_kernel_3(
		unsigned int*lhs_object,
		unsigned int* mmultiplicity,
		int ALPHABET,
		int total_lhs);

__global__ void make_partition_phase_1_kernel_3(int* partition,
		unsigned int* rules_size,
		unsigned int*lhs_object,
		unsigned int* mmultiplicity,
		int * alphabet,
		int NUM_RULES,
		int ALPHABET,
		int num_membranes);
__global__ void make_partition_phase_1_5_kernel_3(int* partition,
		unsigned int* rules_size,
		unsigned int*lhs_object,
		unsigned int* mmultiplicity,
		int * alphabet,
		int NUM_RULES,
		int ALPHABET);
__global__ void make_partition_phase_2_kernel_3(int* partition,
		unsigned int* rules_size,
		unsigned int*lhs_object,
		int * alphabet,
		int NUM_RULES,
		int ALPHABET,
		int num_membranes);
__global__ void get_partition_kernel(int* partition,unsigned int* rules_size,
		unsigned int*lhs_object,
		unsigned int* mmultiplicity,
		int * alphabet,
		int NUM_RULES,
		int ALPHABET);

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)


__device__ bool change=true;
/**
 * https://pdfs.semanticscholar.org/4569/2750cb1bf50da6451e70ae06b6519992e4ec.pdf
 * CUDA kernel that computes partition for overlapping rules
 */
__global__ void make_partition_kernel_3(int* partition,
		unsigned int* rules_size,
		unsigned int* lhs_object,
		int * alphabet,
		int NUM_RULES,
		int ALPHABET,
		int num_membranes,
		unsigned int* mmultiplicity) {
	static const int BLOCK_SIZE = 256;
	const int blockCount1 = (NUM_RULES+BLOCK_SIZE-1)/BLOCK_SIZE;
	const int blockCount2 = ((ALPHABET*num_membranes)+BLOCK_SIZE-1)/BLOCK_SIZE;
	change=true;
	bool exit_loop=false;


	while(change || !exit_loop){

		//If nothing changed, we must give a last pass
		exit_loop=!change;
		change=false;
		make_partition_phase_1_kernel_3<<<blockCount1,BLOCK_SIZE>>>( partition, rules_size, lhs_object,mmultiplicity, alphabet, NUM_RULES, ALPHABET, num_membranes);
		make_partition_phase_1_5_kernel_3<<<blockCount1,BLOCK_SIZE>>>( partition, rules_size, lhs_object,mmultiplicity, alphabet, NUM_RULES, ALPHABET);
		make_partition_phase_2_kernel_3<<<blockCount2,BLOCK_SIZE>>>( partition,  rules_size, lhs_object, alphabet, NUM_RULES, ALPHABET, num_membranes);

		cudaDeviceSynchronize();
//		printf("iter:\n");
//		for(int i=0;i<ALPHABET;i++)
//		{
//			printf("%3u, ",i);
//
//		}
//		printf("\n");
//		for(int i=0;i<ALPHABET;i++)
//		{
//			printf("%3u, ",alphabet[i]);
//		}
//		printf("\n");
	}


}
__global__ void make_partition_phase_0_kernel_3(
				unsigned int*lhs_object,
				unsigned int* mmultiplicity,
				int ALPHABET,
				int total_lhs){
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;

	if (idx < total_lhs){
		lhs_object[idx]=lhs_object[idx]+
                GET_MEMBR(mmultiplicity[idx])*ALPHABET;
	}
}

__global__ void make_partition_phase_1_kernel_3(int* partition,
		unsigned int* rules_size,
		unsigned int*lhs_object,
		unsigned int* mmultiplicity,
		int * alphabet,
		int NUM_RULES,
		int ALPHABET,
		int num_membranes) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;

	if (idx < NUM_RULES){

		unsigned rule_id_begin=rules_size[idx];
		unsigned rule_id_end=rules_size[idx+1];
		int val=ALPHABET*num_membranes;

		for (unsigned int k=rule_id_begin; k<rule_id_end; k++){

			val=min(val,alphabet[lhs_object[k]]);
		}

		partition[idx]=val;

	}
}
__global__ void make_partition_phase_1_5_kernel_3(int* partition,
		unsigned int* rules_size,
		unsigned int*lhs_object,
		unsigned int* mmultiplicity,
		int * alphabet,
		int NUM_RULES,
		int ALPHABET) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;

	if (idx < NUM_RULES){

		unsigned rule_id_begin=rules_size[idx];
		unsigned rule_id_end=rules_size[idx+1];

		unsigned val=partition[idx];

		for (unsigned int j=rule_id_begin; j<rule_id_end; j++){
			int old_val=atomicMin((alphabet+lhs_object[j]),val);
			if(old_val!=val)
				change=true;
		}


	}
}

//Pointer jumping
__global__ void make_partition_phase_2_kernel_3(int* partition,
		unsigned int* rules_size,
		unsigned int*lhs_object,
		int * alphabet,
		int NUM_RULES,
		int ALPHABET,
		int num_membranes) {

	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < ALPHABET*num_membranes){
		int i =alphabet[idx];
		int i_1 =alphabet[i];
		if(i_1!=i){
			//We must do another loop
			change=true;
		}
		while(i_1!=i)
		{
			i=i_1;
			i_1=alphabet[i];
		}
		atomicMin(alphabet+idx,i_1);

	}
}


__global__ void get_partition_kernel(int* partition,
		unsigned int* rules_size,
		unsigned int*lhs_object,
		unsigned int* mmultiplicity,
		int * alphabet,
		int NUM_RULES,
		int ALPHABET) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < NUM_RULES){

		partition[idx]=alphabet[lhs_object[rules_size[idx]]];
		//printf("set rule %i %i with object %u and partition %i \n",idx,rules_size[idx],lhs_object[rules_size[idx]],partition[idx]);

	}
}


/**
 * Host function that copies the data and launches the work on GPU
 */
void make_partition_gpu(int* partition,
		unsigned int* rules_size,
		unsigned int*lhs_object,
		int * alphabet,
		int num_rules,
		int num_objects,
		int num_membranes,

		unsigned int* mmultiplicity,
		int mult_size){
	int * d_partition;
	unsigned int * d_rules_size;
	unsigned int * d_lhs_object;

	unsigned int * d_mmultiplicity;
	int * d_alphabet;
	int total_lhs=rules_size[num_rules];


	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_partition, sizeof(int)*num_rules));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_rules_size,sizeof(int)*(num_rules+1)));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_lhs_object, sizeof(unsigned int)*total_lhs));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_alphabet, sizeof(int)*num_objects*num_membranes));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_mmultiplicity, sizeof(MULTIPLICITY)*mult_size));

	CUDA_CHECK_RETURN(cudaMemcpy(d_partition, partition, sizeof(int)*num_rules, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_rules_size, rules_size, sizeof(int)*(num_rules+1), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_lhs_object, lhs_object,sizeof(unsigned int)*total_lhs, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_alphabet, alphabet, sizeof(int)*num_objects*num_membranes, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_mmultiplicity, mmultiplicity, sizeof(MULTIPLICITY)*mult_size, cudaMemcpyHostToDevice));

	static const int BLOCK_SIZE = 256;
	const int blockCount = (total_lhs+BLOCK_SIZE-1)/BLOCK_SIZE;
	const int blockCount1 = (num_rules+BLOCK_SIZE-1)/BLOCK_SIZE;


	make_partition_phase_0_kernel_3<<<blockCount,BLOCK_SIZE>>>(
			d_lhs_object,
			d_mmultiplicity,
			num_objects,
			total_lhs);
    make_partition_kernel_3<<<1,1>>>(d_partition,
    		d_rules_size,
    		d_lhs_object,
    		d_alphabet,
    		num_rules,
    		num_objects,
    		num_membranes,
    		d_mmultiplicity);
	get_partition_kernel<<<blockCount1, BLOCK_SIZE>>> (
			d_partition,
			d_rules_size,
			d_lhs_object,
			d_mmultiplicity,
			d_alphabet,
			num_rules,
			num_objects);

    cudaDeviceSynchronize();


	CUDA_CHECK_RETURN(cudaMemcpy(partition, d_partition, sizeof(int)*num_rules, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(alphabet, d_alphabet, sizeof(int)*num_objects*num_membranes, cudaMemcpyDeviceToHost));


	CUDA_CHECK_RETURN(cudaFree(d_partition));
	CUDA_CHECK_RETURN(cudaFree(d_rules_size));
	CUDA_CHECK_RETURN(cudaFree(d_lhs_object));
	CUDA_CHECK_RETURN(cudaFree(d_alphabet));
	CUDA_CHECK_RETURN(cudaFree(d_mmultiplicity));
}

int initialize_rules(int *data, int size,int max_objects_per_rule)
{
	data[0]=0;
	for (int i = 1; i < size; ++i){

		data[i] =data[i-1]+(rand()%max_objects_per_rule)+1;
	}
	return data[size-1];
}
void initialize_lhs(int *data, int size,int num_rules,int num_objects)
{
	for (int i = 0; i < size; ++i){
		data[i] =rand()%num_objects;
	}

}

/**
 * Modified sequential version of partition kernel version 3
 * It works
 */
void make_partition(int* partition, unsigned int* rules_size,unsigned int*lhs_object, int * alphabet,int num_rules,int num_objects){
	bool change=true;
	bool exit_loop=false;

	while(change || !exit_loop){

		//If nothing changed, we must give a last pass
		exit_loop=!change;
		change=false;

		for(int i=0;i<num_rules;i++){
			for(int j=i+1;j<num_rules;j++){
				if(partition[j]!=partition[i] && check_compete(i,j,rules_size,lhs_object)){
					if(partition[j]<partition[i]){
						partition[i]=partition[j];
						change=true;
					}else{
						partition[j]=partition[i];
					}
				}
			}
		}
		for(int idx=0;idx<num_rules;idx++){
			int i =partition[idx];
			int i_1 =partition[i];
			bool must_go_on=false;
			while(i_1!=i)
			{
				must_go_on=true;
				i=i_1;
				i_1=partition[i];
			}
			partition[idx]=min(partition[idx],i_1);
			if(must_go_on){
				change=true;
			}
		}
		}
}

/**
 * Modified sequential version of partition kernel version 3
 * It works
 */
void make_partition(int* partition, unsigned int* rules_size,unsigned int*lhs_object,int * alphabet,int num_rules,int num_objects,unsigned int *membrane, unsigned int* mmultiplicity){
	bool change=true;
	bool exit_loop=false;

	while(change || !exit_loop){

		//If nothing changed, we must give a last pass
		exit_loop=!change;
		change=false;

		for(int i=0;i<num_rules;i++){
			for(int j=i+1;j<num_rules;j++){
				if(partition[j]!=partition[i] && check_compete(i,j,rules_size,lhs_object,membrane,mmultiplicity)){
					if(partition[j]<partition[i]){
						partition[i]=partition[j];
						change=true;
					}else{
						partition[j]=partition[i];
					}
				}
			}
		}
		for(int idx=0;idx<num_rules;idx++){
			int i =partition[idx];
			int i_1 =partition[i];
			bool must_go_on=false;
			while(i_1!=i)
			{
				must_go_on=true;
				i=i_1;
				i_1=partition[i];
			}
			partition[idx]=min(partition[idx],i_1);
			if(must_go_on){
				change=true;
			}
		}
		}
}



bool check_compete(int block_a,int block_b,unsigned int* rules_size,unsigned int * lhs_object){
	bool res=false;
	for (unsigned int j=rules_size[block_a]; j<rules_size[block_a+1]; j++){
		for (unsigned int k=rules_size[block_b]; k<rules_size[block_b+1]; k++){

			if(lhs_object[j]==lhs_object[k]){
				res=true;
				break;
			}
		}
	}
	return res;
}

bool check_compete(int block_a,int block_b,unsigned int* rules_size,unsigned int * lhs_object,unsigned int *membrane, unsigned int* mmultiplicity){
	bool res=false;
	for (unsigned int j=rules_size[block_a]; j<rules_size[block_a+1]; j++){
		for (unsigned int k=rules_size[block_b]; k<rules_size[block_b+1]; k++){

			//If they share an object in the same membrane
			if(lhs_object[j]==lhs_object[k]
		         &&GET_MEMBR(mmultiplicity[j])==GET_MEMBR(mmultiplicity[k])){

					// Also the blocks stand for different membranes OR
					// They stand for the same membrane and have the same charge
					//if(GET_MEMBRANE(membrane[block_a])!=GET_MEMBRANE(membrane[block_b])
					//		|| (GET_ALPHA(membrane[block_a])==GET_ALPHA(membrane[block_b]))){
						res=true;
						break;

					//}
			}
		}
	}
	return res;
}

void reset_partition(int* partition,int* alphabet,int num_rules,int num_objects) {
	for (int i = 0; i < num_objects; i++) {
		alphabet[i] = i;
	}
	//At most, they will be all independent
	for (int i = 0; i < num_rules; i++) {
		partition[i] = i;
	}
}

void print_header(int num_rules,int num_objects,int max_objects_per_rule){
	std::cout<< "--- " << num_rules <<" rules generated with at most "
			<< max_objects_per_rule<< " objects each and "
			<<num_objects <<" objects in alphabet" <<" ---"<< std::endl;

}
void print_rules(unsigned int* rules_size,unsigned int* lhs_object,int num_rules,int num_objects) {

	for (int i = 0; i < num_rules; i++) {
		std::cout << "Rule " << i << std::endl;
		for (int j = rules_size[i]; j < rules_size[i + 1]; j++) {
			std::cout << "\t Object " << lhs_object[j] << std::endl;
		}
	}
}

void print_partition( int* partition, int* alphabet,int num_rules,int num_objects) {


	for (int i = 0; i < num_rules; i++) {
		std::cout << "\t Rule " << i << " belongs to part " << partition[i]
				<< std::endl;
	}
//	for (int i = 0; i < ALPHABET; i++) {
//		std::cout << "\t Object " << i << " belongs to part " << alphabet[i]
//				<< std::endl;
//	}
}
void print_comparing_partition(int* partition, int* alphabet,int* partition2, int* alphabet2,int num_rules,int num_objects) {

	for (int i = 0; i < num_rules; i++) {
		std::cout << "\t Rule " << i << " belongs to part: " << partition[i]
		          << "\t | \t " << partition2[i]
				<< std::endl;
	}
//	for (int i = 0; i < ALPHABET; i++) {
//		std::cout << "\t Object " << i << " belongs to part " << alphabet[i]
//				<< "\t | \t " << alphabet2[i]
//				<< std::endl;
//	}
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

void compare_partition(int* partition, int* alphabet,int* partition2, int* alphabet2,int num_rules,int num_objects){

//	for(int i=0;i<ALPHABET;i++){
//		if(alphabet[i]!=alphabet2[i]){
//			std::cout<<"Alphabet not matching"<<std::endl;
//			break;
//		}
//	}

	for(int i=0;i<num_rules;i++){
		if(partition[i]!=partition2[i]){
			std::cout<<"Partition not matching"<<std::endl;
			return;
		}
	}

	std::cout<<"Works great!!!"<<std::endl;
}
int normalize_partition(int* partition, int* trans_partition,int size){

	for(int i=0;i<size;i++){
		trans_partition[i]=-1;
	}

	int part_index=0;
	for (int i=0;i<size;i++){
		for(int j=0;j<i;j++){
			if(partition[j]==partition[i]){
				trans_partition[i]=trans_partition[j];
				break;
			}
		}
		if(trans_partition[i]==-1){
			trans_partition[i]=part_index;
			part_index++;
		}
	}

	return part_index;
}
void initialize_partition_structures(int* partition,int num_partitions,int* rules_size,int num_rules,int *lhs_object){

	Options* opt=new Options[num_partitions];

	for(int i=0;i<num_partitions;i++){
		//A struct for each partition
		opt[i]=new struct _options;
	}
	for(int i=0;i<num_rules;i++){
		opt[partition[i]]->num_rule_blocks++;
	}

}


}
