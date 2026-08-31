#include <stdio.h>

typedef int (*CMP)(const void*,const void*);          
int min(const void* a, const void* b) { 
    return *(int*)a<*(int*)b?*(int*)a:*(int*)b; 
};
int max(const void* a, const void* b) { 
    return (int)((int*)a>(int*)b?(int*)a:(int*)b); 
};

int compare(CMP f, int a, int b)
{
  return f(&a, &b);
} 


int main(void) {
    printf("min=%d\n", compare(min, 3, 5));
    printf("max=%d\n", compare(max, 3, 5));

    return 0;
}