#include <cuda.h>
#include <iostream>
#include <vector>

struct DataSet{
    ~DataSet(){ delete [] flatData; }
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

int main(){
    DataSet data = createTestDataSet();
    print(data);
}