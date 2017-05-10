#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#include "immintrin.h"

#include <smmintrin.h>
#include <xmmintrin.h>
#include <pmmintrin.h>

#include <iostream>
#include <set>

//#define sequential
#define vectorized

#define MIN(a,b) (((a)<(b))?(a):(b))

void inspector(int cur_tid, int cur_i);
void executor(int index, int cur_tid, int cur_i);

int no_of_nodes;
int edge_list_size;
FILE *fp;

// Structure to hold a node information
struct Node {
    int starting;
    int no_of_edges;
};

void BFSGraph(int argc, char **argv);

void Usage(int argc, char **argv) {

    fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
}

// Setup aligned arrays
Node *h_graph_nodes = (Node *)malloc(sizeof(Node) * 1048576);
__attribute__((aligned(64))) int *h_graph_mask, *h_updating_graph_mask, *h_graph_visited, *h_graph_edges, *h_cost;

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    no_of_nodes = 0;
    edge_list_size = 0;

    h_graph_mask = (int *)_mm_malloc(sizeof(int) * 1048576, 64);
    h_updating_graph_mask = (int *)_mm_malloc(sizeof(int) * 1048576, 64);
    h_graph_visited = (int *)_mm_malloc(sizeof(int) * 1048576, 64);
    h_graph_edges = (int *)_mm_malloc(sizeof(int) * 6292610, 64);
    h_cost = (int *)_mm_malloc(sizeof(int) * 1048576, 64);

    assert(h_graph_mask && h_updating_graph_mask && h_graph_visited && h_graph_edges && h_cost);

    memset(h_graph_mask, 0, sizeof(int) * 1048576);
    memset(h_updating_graph_mask, 0, sizeof(int) * 1048576);
    memset(h_graph_visited, 0, sizeof(int) * 1048576);
    memset(h_graph_edges, 0, sizeof(int) * 6292610);
    memset(h_cost, 0, sizeof(int) * 1048576);

    clock_t t1,t2;
    t1 = clock();
    BFSGraph(argc, argv);
    t2 = clock();
    float diff = ((float)t2-(float)t1);
    //printf("Program completed in %f clock ticks. \n", diff);
}

//int *idBuffer = (int *)malloc(sizeof(int) * 1048576); // Use 128 as arbitrary size of buffer
//int *tidBuffer = (int *)malloc(sizeof(int) * 1048576);



////////////////////////////////////////////////////////////////////////////////
// Apply BFS on a Graph
////////////////////////////////////////////////////////////////////////////////
void BFSGraph(int argc, char **argv) {
    char *input_f;

    if (argc != 2) {
        Usage(argc, argv);
        exit(0);
    }

    input_f = argv[1];

    printf("Reading File\n");
    // Read in Graph from a file
    fp = fopen(input_f, "r");
    if (!fp) {
        printf("Error Reading graph file\n");
        return;
    }

    int source = 0;

    fscanf(fp, "%d", &no_of_nodes);

    // allocate host memory
    //Node *h_graph_nodes = (Node *)malloc(sizeof(Node) * no_of_nodes);
    //int *h_graph_mask = (int *)malloc(sizeof(int) * no_of_nodes);
    //int *h_updating_graph_mask = (int *)malloc(sizeof(int) * no_of_nodes);
    //int *h_graph_visited = (int *)malloc(sizeof(int) * no_of_nodes);

    int start, edgeno;
    // initalize the memory
    for (unsigned int i = 0; i < no_of_nodes; i++)
    {
        fscanf(fp, "%d %d", &start, &edgeno);
        h_graph_nodes[i].starting = start;
        h_graph_nodes[i].no_of_edges = edgeno;
        h_graph_mask[i] = 0;
        h_updating_graph_mask[i] = 0;
        h_graph_visited[i] = 0;
    }

    // read the source node from the file
    fscanf(fp, "%d", &source);
    // source=0; //tesing code line

    // set the source node as true in the mask
    h_graph_mask[source] = 1;
    h_graph_visited[source] = 1;

    fscanf(fp, "%d", &edge_list_size);

    int id, cost;
    //int *h_graph_edges = (int *)malloc(sizeof(int) * edge_list_size);
    for (int i = 0; i < edge_list_size; i++)
    {
        fscanf(fp, "%d", &id);
        fscanf(fp, "%d", &cost);
        h_graph_edges[i] = id;
    }

    if (fp)
        fclose(fp);


    // allocate mem for the result on host side
    //int *h_cost = (int *)malloc(sizeof(int) * no_of_nodes);
    for (int i = 0; i < no_of_nodes; i++)
        h_cost[i] = -1;
    h_cost[source] = 0;

    printf("Start traversing the tree\n");

    int k = 0;

    //////////////////////////////////////////

    //int cur_tid, cur_i;

    int *idBuffer = (int *)malloc(sizeof(int) * 1186503); // Use 128 as arbitrary size of buffer 1048576
    int *tidBuffer = (int *)malloc(sizeof(int) * 1186503);

    float diff = 0.0;

    bool stop;
    do {
        // if no thread changes this value then the loop stops
        stop = false;

///////////////////////////////////////////// Sequential
#ifdef vectorized

        //int *idBuffer = (int *)malloc(sizeof(int) * 1186503); // Use 128 as arbitrary size of buffer 1048576
        //int *tidBuffer = (int *)malloc(sizeof(int) * 1186503);
        clock_t t1,t2;

        int index = 0;
        // Inspector
        for (int tid = 0; tid < no_of_nodes; tid++) {
            if (h_graph_mask[tid] == 1) {
                h_graph_mask[tid] = 0;
                for (int i = h_graph_nodes[tid].starting;
                     i < (h_graph_nodes[tid].no_of_edges +
                          h_graph_nodes[tid].starting);
                     i++) {
                    int id = h_graph_edges[i];
                    if (!h_graph_visited[id]) {
                        idBuffer[index] = id;
                        tidBuffer[index] = tid;
                        //h_cost[id] = h_cost[tid] + 1;
                        //h_updating_graph_mask[id] = true;
                        index++;
                    }
                }
            }
        }

        __m512i one_vec = _mm512_set1_epi32(1); // Contains 16 ones
        __m512i zero_vec = _mm512_setzero_epi32(); // Contains 16 zeroes
        __m512i id_index_vec, tid_index_vec;
        __m512i gather, id_conflict, tid_conflict;
        __m512i h_updating_graph_mask_vec;
        __mmask16 mask, idConflictMask, tidConflictMask;

        /*int _bi = 0;
        //#pragma vector aligned
        for (; _bi + 16 <= index; _bi += 16){
            //No conflict, SIMD
            if (test){

            } else {
                for (){

                }
            }

        }

        //for the rest
        for (int i = _bi; i < index; ++i){


        }*/

	// Buffer reordering on idBuffer
	int buffLen = 1186503;
	int tmp;
	std::set<int> sett;
	for (int i = 0; i < buffLen; i++)
	{
		if (i % 16 == 0)
		{
			printf("i = %d\n", i);
			sett.clear();
			for (int j = i; j < MIN(i+16, buffLen); j++)
			{
				bool found1 = sett.count(idBuffer[j]) != 0;
				if (found1)
				{
					for (int k = i + 16; k < buffLen; k++)
					{
						bool found2 = sett.count(idBuffer[k]) != 0;
						if (!found2)
						{
							printf("swapping\n");
							sett.insert(idBuffer[k]);
							//swap idBuffer[j] and idBuffer[k]
							tmp = idBuffer[j];
							idBuffer[j] = idBuffer[k];
							idBuffer[k] = tmp;
							break;
						}
					}
				}
				else
				{
					sett.insert(idBuffer[j]);
				}
			}
		}
	}
	
	// end buffer reordering

        t1 = clock();
        for (int i = 0; i < index;)
        {
    		if (i + 16 <= index) // Vectorized
    		{
                id_index_vec = _mm512_load_epi32(&idBuffer[i]);
                tid_index_vec = _mm512_load_epi32(&tidBuffer[i]);
                id_conflict = _mm512_conflict_epi32(id_index_vec);
                //tid_conflict = _mm512_conflict_epi32(tid_index_vec);
                idConflictMask = _mm512_cmpeq_epi32_mask(id_conflict, zero_vec);
                //tidConflictMask = _mm512_cmpeq_epi32_mask(tid_conflict, zero_vec);

                if (idConflictMask != 0x0000)
                {
                    h_cost[idBuffer[i]] = h_cost[tidBuffer[i]] + 1;
                    h_updating_graph_mask[idBuffer[i]] = 1;
                    i = i + 1;
                }
                else
                {
                    id_index_vec = _mm512_load_epi32(&idBuffer[i]);
                    tid_index_vec = _mm512_load_epi32(&tidBuffer[i]);
                    gather = _mm512_i32gather_epi32(tid_index_vec, &h_cost[i], 4); // gather index values
                    gather = _mm512_add_epi32(gather, one_vec); // add one to every element in gather
                    _mm512_i32scatter_epi32(&h_cost[i], id_index_vec, gather, 4); // scatter elements back into h_cost
        			_mm512_i32scatter_epi32(&h_updating_graph_mask[i], id_index_vec, one_vec, 4); // // scatter elements into h_updating_graph_mask
        			i = i + 16;
                }
    		}
    		else // sequential
    		{
    			h_cost[idBuffer[i]] = h_cost[tidBuffer[i]] + 1;
    			h_updating_graph_mask[idBuffer[i]] = 1;
    			i = i + 1;
    		}
        }
        printf("index: %d\n", index);

        for (int tid = 0; tid < no_of_nodes;) {
            if (tid + 16 <= no_of_nodes) // Can be vectorized
            {
                h_updating_graph_mask_vec = _mm512_load_epi32(&h_updating_graph_mask[tid]);
                mask = _mm512_cmpeq_epi32_mask(h_updating_graph_mask_vec, one_vec);

                _mm512_mask_store_epi32(&h_graph_mask[tid], mask, one_vec);
                _mm512_mask_store_epi32(&h_graph_visited[tid], mask, one_vec);
                _mm512_mask_store_epi32(&h_updating_graph_mask[tid], mask, zero_vec);
                if (mask != 0x0000) // true if any bit in mask is a 1
                    stop = 1;
                tid = tid + 16;
            }
            else // Continue sequentially
            {
                if (h_updating_graph_mask[tid] == 1)
                {
                    h_graph_mask[tid] = 1;
                    h_graph_visited[tid] = 1;
                    stop = 1;
                    h_updating_graph_mask[tid] = 0;
                }
                tid = tid + 1;
            }
        }
        t2 = clock();
        diff = diff + ((float)t2-(float)t1);


        //free(idBuffer);
        //free(tidBuffer);

#endif
//////////////////////////////////// Vectorized

#ifdef sequential


        int *idBuffer = (int *)malloc(sizeof(int) * 1186503); // Use 128 as arbitrary size of buffer 1048576
        int *tidBuffer = (int *)malloc(sizeof(int) * 1186503);

        clock_t t1,t2;

        int index = 0;
        // Inspector
        for (int tid = 0; tid < no_of_nodes; tid++) {
            if (h_graph_mask[tid] == 1) {
                h_graph_mask[tid] = 0;
                for (int i = h_graph_nodes[tid].starting;
                     i < (h_graph_nodes[tid].no_of_edges +
                          h_graph_nodes[tid].starting);
                     i++) {
                    int id = h_graph_edges[i];
                    if (!h_graph_visited[id]) {
                        idBuffer[index] = id;
                        tidBuffer[index] = tid;
                        index++;
                    }
                }
            }
        }

        t1 = clock();
        for (int i = 0; i < index; i++)
        {
            h_cost[idBuffer[i]] = h_cost[tidBuffer[i]] + 1;
            h_updating_graph_mask[idBuffer[i]] = 1;
        }
        printf("index: %d\n", index);

        for (int tid = 0; tid < no_of_nodes; tid++) {
            if (h_updating_graph_mask[tid] == 1) {
                h_graph_mask[tid] = 1;
                h_graph_visited[tid] = 1;
                stop = 1;
                h_updating_graph_mask[tid] = 0;
            }
        }
        t2 = clock();
        diff = diff + ((float)t2-(float)t1);


#endif

        k++;
        //
    } while (stop);

    printf("Program completed in %f clock ticks. \n", diff);


    // Store the result into a file
    if (getenv("OUTPUT")) {
        FILE *fpo = fopen("output.txt", "w");
        for (int i = 0; i < no_of_nodes; i++)
            fprintf(fpo, "%d) cost:%d\n", i, h_cost[i]);
        fclose(fpo);
    }

    // cleanup memory
    free(h_graph_nodes);
    _mm_free(h_graph_edges);
    _mm_free(h_graph_mask);
    _mm_free(h_updating_graph_mask);
    _mm_free(h_graph_visited);
    _mm_free(h_cost);

    free(idBuffer);
    free(tidBuffer);
}


























