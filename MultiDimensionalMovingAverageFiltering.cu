#include <cuda.h>
#include <iostream>
#include <vector>
#include <iomanip>

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
    //~DataSet(){ delete [] flatData; }
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
        std::cout << /*std::setprecision(2) << std::fixed <<*/ data.flatData[i] << "\t";
    }
    std::cout << std::endl << std::endl;
}

typedef dim3 Filter;

__device__
int getGlobalIdx_3D_3D(){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
    + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
    + (threadIdx.z * (blockDim.x * blockDim.y))
    + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

__global__ void MovingAverageKernel(DataSet input, Filter filter, DataSet output){
    uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint64_t idy = blockDim.y * blockIdx.y + threadIdx.y;
    uint64_t idz = blockDim.z * blockIdx.z + threadIdx.z;

    uint64_t idglobal = getGlobalIdx_3D_3D();

    if (idglobal < output.flatDataSize && 
        idx <= output.dimension.x &&
        idy <= output.dimension.y &&
        idz <= output.dimension.z 
    ){
        float sum = 0;
        if (idglobal == 69)
        printf("Output 0 = (");
        for (uint64_t z = 0; z < filter.z; z++)
            for (uint64_t y = 0; y < filter.y; y++)
                for (uint64_t x = 0; x < filter.x; x++) {
                    unsigned int iddd = idx+x+ input.dimension.x * ((idy+y) + input.dimension.y*(idz + z));
                    sum += input.flatData[iddd];
                    if (idglobal == 69)
                        printf(" %f [%d] + \n", input.flatData[iddd], iddd);
                }
        sum /= (float)(filter.x * filter.y * filter.z);
        if (idglobal == 69)
            printf(" ) / %f = %f", (float)filter.x * filter.y * filter.z, sum);
        output.flatData[idglobal]=sum;
    }
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

    dim3 threadsperblock{ 1, 1, 1 };
    dim3 blocksneeded = {
        device_output.dimension.x / threadsperblock.x + 1,
        device_output.dimension.y / threadsperblock.y + 1,
        device_output.dimension.z / threadsperblock.z + 1
    };
    MovingAverageKernel<<< blocksneeded, threadsperblock >>>(device_input, filter, device_output);
    gpuErrchk(cudaMemcpy(output.flatData, device_output.flatData, output.flatDataSize*sizeof(float), cudaMemcpyDeviceToHost));

    return std::move(output);
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