#include <cuda.h>
#include <iostream>
#include <vector>

struct DataSet{
    dim3         dimension;
    float*       flatData;
    unsigned int flatDataSize;
};

typedef dim3 Filter;

__global__ void MovingAverage(DataSet input, Filter filter, DataSet output){

}

int main(){
    DataSet data;
    std::vector<float> input;
}