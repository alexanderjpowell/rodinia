#include <stdio.h>
#include <string.h>
#include "immintrin.h"
#include <stdlib.h>
#include <assert.h>

__attribute__((aligned(64))) int *test;
//test = (int *)_mm_malloc(sizeof(int) * 10, 64);
//assert(test);
//memset(test, 0, 10 * sizeof(int));

int main()
{
	test = (int *)_mm_malloc(sizeof(int) * 10, 64);
	printf("testing\n");
	return 0;
}
