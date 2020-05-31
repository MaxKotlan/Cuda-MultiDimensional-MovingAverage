#include <cuda.h>
#include <iostream>
#include <vector>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


struct DataSet{
    ~DataSet(){ delete [] flatData; }
    DataSet(){};
    DataSet(unsigned int x, unsigned int y, unsigned int z){
        dimension = {x, y, z};
        flatDataSize = x*y*z;
        flatData = new float[flatDataSize];
    }
    dim3         dimension;
    float*       flatData;
    unsigned int flatDataSize;
};

void print(DataSet &data){
    for (int i = 0; i < data.flatDataSize; i++){
        if (i%data.dimension.x == 0) std::cout << std::endl;
        if (i%(data.dimension.x*data.dimension.y) == 0) std::cout << std::endl;
        std::cout << data.flatData[i] << "\t";
    }
    std::cout << std::endl << std::endl;
}

typedef dim3 Filter;

__global__ void MovingAverage(DataSet input, Filter filter, DataSet output){
    
}

DataSet createTestDataSet(){
    DataSet d;
    d.dimension = { 8, 6, 3 };
    d.flatDataSize = d.dimension.x*d.dimension.y*d.dimension.z;
    d.flatData = new float[d.flatDataSize]{ 
        1.1f, 2.1f, 3.1f, 4.1f, 5.1f, 6.1f, 7.1f, 8.1f,
        1.1f, 2.1f, 3.1f, 4.1f, 5.1f, 6.1f, 7.1f, 8.1f,
        1.1f, 2.1f, 3.1f, 4.1f, 5.1f, 6.1f, 7.1f, 8.1f,
        1.1f, 2.1f, 3.1f, 4.1f, 5.1f, 6.1f, 7.1f, 8.1f,
        1.1f, 2.1f, 3.1f, 4.1f, 5.1f, 6.1f, 7.1f, 8.1f,
        1.1f, 2.1f, 3.1f, 4.1f, 5.1f, 6.1f, 7.1f, 8.1f,
        1.2f, 2.2f, 3.2f, 4.2f, 5.2f, 6.2f, 7.2f, 8.2f,
        1.2f, 2.2f, 3.2f, 4.2f, 5.2f, 6.2f, 7.2f, 8.2f,
        1.2f, 2.2f, 3.2f, 4.2f, 5.2f, 6.2f, 7.2f, 8.2f,
        1.2f, 2.2f, 3.2f, 4.2f, 5.2f, 6.2f, 7.2f, 8.2f,
        1.2f, 2.2f, 3.2f, 4.2f, 5.2f, 6.2f, 7.2f, 8.2f,
        1.2f, 2.2f, 3.2f, 4.2f, 5.2f, 6.2f, 7.2f, 8.2f,
        1.3f, 2.3f, 3.3f, 4.3f, 5.3f, 6.3f, 7.3f, 8.3f,
        1.3f, 2.3f, 3.3f, 4.3f, 5.3f, 6.3f, 7.3f, 8.3f,
        1.3f, 2.3f, 3.3f, 4.3f, 5.3f, 6.3f, 7.3f, 8.3f,
        1.3f, 2.3f, 3.3f, 4.3f, 5.3f, 6.3f, 7.3f, 8.3f,
        1.3f, 2.3f, 3.3f, 4.3f, 5.3f, 6.3f, 7.3f, 8.3f,
        1.3f, 2.3f, 3.3f, 4.3f, 5.3f, 6.3f, 7.3f, 8.3f
    };
    return d;
}

DataSet MovingAverage(DataSet &input, Filter &filter){
    /*Initalize output dataset using the size of the input and the filter*/
    DataSet output(
        input.dimension.x - filter.x + 1,
        input.dimension.y - filter.y + 1,
        input.dimension.z - filter.z + 1
    );

    /*Initalize data on the device*/
    DataSet device_input;
    device_input.dimension    = input.dimension;
    device_input.flatData     = nullptr;
    device_input.flatDataSize = input.flatDataSize;

    DataSet device_output;
    device_output.dimension    = output.dimension;
    device_output.flatData     = nullptr;
    device_output.flatDataSize = output.flatDataSize;

    gpuErrchk(cudaMalloc((void **)&device_input.flatData,  sizeof(float)*device_input.flatDataSize));
    gpuErrchk(cudaMalloc((void **)&device_output.flatData, sizeof(float)*device_output.flatDataSize));
    gpuErrchk(cudaMemcpy(device_input.flatData, input.flatData, sizeof(float)*device_input.flatDataSize, cudaMemcpyHostToDevice));

    return output;
}


int main(){
    DataSet input = createTestDataSet();
    std::cout << "Input DataSet: " << std::endl;
    print(input);
    std::cout << "==========seperate line ==========" << std::endl;

    Filter filter{2,2,2};
    DataSet output = MovingAverage(input, filter);
    print(output);
}