// Adapted from: 
// https://github.com/Liu-xiandong/How_to_optimize_in_GPU/blob/master/spmv/spmv.cu

#include <bits/stdc++.h>
#include <cuda.h>
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime_api.h> 
#include <cusparse.h> 

using namespace std;

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

void add(int a, int b, float c,
    int *h, int *e, int *ne, float *w, int &idx)
{
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx++;
}

// get basic information of the matrix
void readVerEdges(int &is_weighted, int &n, int &t, int &m, std::string &file)
{
    std::ifstream input;
    input.open("matrix/" + file + ".mtx");

    while (input.peek() == '%')
        input.ignore(2048, '\n');

    input >> n >> t >> m;

    std::string str;
    input.ignore();
    getline(input, str);
    int cnt =0;
    for(auto c:str){
        if(c==' '){
            cnt++;
        }
    }
    if(cnt == 1){
        is_weighted = 0;
    }
    else if(cnt == 2){
        is_weighted = 1;
    }
    else{
        std::cout<<"error! you need to get right mtx input\n";
        exit(0);
    }
    input.close();
}

// read matrix and store into CSR form
void readMtxFile(int is_weighted, int n, int m,
            int *row_offset, int *col_index, float *val,
            std::string &file)
{
    ifstream input;
    input.open("matrix/" + file + ".mtx");

    while (input.peek() == '%')
        input.ignore(2048, '\n');

    int t;
    input >> n >> t >> m;
    int *h = (int *)malloc((n + 10) * sizeof(int));
    memset(h, -1, sizeof(int) * (n + 10));
    int *e = (int *)malloc((m + 10) * sizeof(int));
    int *ne = (int *)malloc((m + 10) * sizeof(int));
    float *w = (float *)malloc((m + 10) * sizeof(float));
    int idx = 0;

    int a, b;
    double c;
    srand((int)time(0));
    if(is_weighted == 0){
        while (input >> a >> b)
        {
            a--;
            b--;
            c = a%13;
            float tc = static_cast<float>(c);
            add(a, b, tc, h, e, ne, w, idx);
        }
    }
    else if(is_weighted == 1){
        while (input >> a >> b >> c)
        {
            a--;
            b--;
            float tc = static_cast<float>(c);
            add(a, b, tc, h, e, ne, w, idx);
        }
    }
    else{
        std::cout<<"error! you need to get right mtx input\n";
        exit(0);
    }
    

    row_offset[0] = 0;
    int nnz_num = 0;

    for (int i = 0; i < n; i++)
    {
        int count = 0;
        for (int j = h[i]; j != -1; j = ne[j])
        {
            count++;
            int nextNode = e[j];
            float nextWeight = w[j];
            col_index[nnz_num] = nextNode;
            val[nnz_num] = nextWeight;
            nnz_num++;
        }
        row_offset[i + 1] = row_offset[i] + count;
    }

    input.close();
    free(h);
    free(e);
    free(ne);
    free(w);
}

// device end tool
template <unsigned int WarpSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    // IMPORTANT: for sub-warp reductions (WarpSize < 32), constrain shuffle width
    // to avoid cross-vector contamination.
    if (WarpSize >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16, WarpSize);
    if (WarpSize >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8,  WarpSize);
    if (WarpSize >= 8)  sum += __shfl_down_sync(0xffffffff, sum, 4,  WarpSize);
    if (WarpSize >= 4)  sum += __shfl_down_sync(0xffffffff, sum, 2,  WarpSize);
    if (WarpSize >= 2)  sum += __shfl_down_sync(0xffffffff, sum, 1,  WarpSize);
    return sum;
}

// core GPU kernel
template <typename IndexType, typename ValueType, unsigned int VECTORS_PER_BLOCK, unsigned int THREADS_PER_VECTOR>
__global__ void My_spmv_csr_kernel(const IndexType row_num,
                       const IndexType * A_row_offset,
                       const IndexType * A_col_index,
                       const ValueType * A_value,
                       const ValueType * x,
                       ValueType * y)
{
    const IndexType THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;
    const IndexType thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
    const IndexType thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const IndexType row_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index

    if(row_id < row_num){
        const IndexType row_start = A_row_offset[row_id];                  //same as: row_start = Ap[row];
        const IndexType row_end   = A_row_offset[row_id+1];

        // initialize local sum
        ValueType sum = 0;

        // accumulate local sums
        for(IndexType jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR)
            sum += A_value[jj] * x[ A_col_index[jj] ];

        sum = warpReduceSum<THREADS_PER_VECTOR>(sum);
        if (thread_lane == 0){
            y[row_id] = sum;
        }   
    }
}

// Bucketed CSR SpMV: each "vector" processes one row, but rows are provided by an index list.
template <typename IndexType, typename ValueType, unsigned int VECTORS_PER_BLOCK, unsigned int THREADS_PER_VECTOR>
__global__ void My_spmv_csr_bucket_kernel(const IndexType bucket_row_num,
                       const IndexType * __restrict__ row_ids,
                       const IndexType * __restrict__ A_row_offset,
                       const IndexType * __restrict__ A_col_index,
                       const ValueType * __restrict__ A_value,
                       const ValueType * __restrict__ x,
                       ValueType * __restrict__ y)
{
    const IndexType THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;
    const IndexType vector_id   = (THREADS_PER_BLOCK * blockIdx.x + threadIdx.x) / THREADS_PER_VECTOR;
    const IndexType thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);

    if (vector_id < bucket_row_num) {
        const IndexType row_id = row_ids[vector_id];
        const IndexType row_start = A_row_offset[row_id];
        const IndexType row_end   = A_row_offset[row_id + 1];

        ValueType sum = 0;
        for (IndexType jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR) {
            sum += A_value[jj] * x[A_col_index[jj]];
        }

        sum = warpReduceSum<THREADS_PER_VECTOR>(sum);
        if (thread_lane == 0) {
            y[row_id] = sum;
        }
    }
}

// Long-row kernel: one CTA (block) per row to better utilize threads on very long rows.
template <int BLOCK_SIZE>
__global__ void My_spmv_csr_block_per_row_kernel(const int bucket_row_num,
                       const int * __restrict__ row_ids,
                       const int * __restrict__ A_row_offset,
                       const int * __restrict__ A_col_index,
                       const float * __restrict__ A_value,
                       const float * __restrict__ x,
                       float * __restrict__ y)
{
    const int bucket_row_idx = blockIdx.x;
    if (bucket_row_idx >= bucket_row_num) return;

    const int row_id = row_ids[bucket_row_idx];
    const int row_start = A_row_offset[row_id];
    const int row_end   = A_row_offset[row_id + 1];

    float thread_sum = 0.0f;
    for (int jj = row_start + threadIdx.x; jj < row_end; jj += BLOCK_SIZE) {
        thread_sum += A_value[jj] * x[A_col_index[jj]];
    }

    // Block reduction (BLOCK_SIZE must be multiple of 32)
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    float warp_sum = warpReduceSum<32>(thread_sum);

    __shared__ float warp_sums[BLOCK_SIZE / 32];
    if (lane == 0) warp_sums[warp_id] = warp_sum;
    __syncthreads();

    float block_sum = 0.0f;
    if (warp_id == 0) {
        block_sum = (lane < (BLOCK_SIZE / 32)) ? warp_sums[lane] : 0.0f;
        block_sum = warpReduceSum<32>(block_sum);
        if (lane == 0) {
            y[row_id] = block_sum;
        }
    }
}

template<typename T>
void vec_print(vector<T> array){
    for(auto x: array){
        cout<<x<<" ";
    }
    cout<<std::endl;
}

// reference CPU implementation
template <typename IndexType, typename ValueType>
void spmv_cpu_kernel(vector<IndexType> &row_offset,
                vector<IndexType> &col_index,
                vector<ValueType> &value,
                vector<ValueType> &x,
                vector<ValueType> &y,
                IndexType row_num)
{
    for(int i=0; i<row_num; i++){
        ValueType res = 0;
        IndexType num = row_offset[i+1] - row_offset[i];
        for(int j=0; j<num; j++){
            IndexType index = row_offset[i] + j;
            res += value[index]*x[col_index[index]];
        }
        y[i] = res;
    }
}


int main(int argc, char **argv)
{
    if (argc != 3) {
        printf("usage: ./spmv -f [matrix]\n");
        exit(0);
    }
    string file;
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-f") == 0)
        {
            file = argv[i + 1];
        }
    }

    // read mtx file and convert to csr
    int is_weighted = -1;
    int row_num;
    int col_num;
    int nnz_num;
    readVerEdges(is_weighted, row_num, col_num, nnz_num, file);
    vector<int> row_offset(row_num + 1);
    vector<int> col_index(nnz_num);
    vector<float> value(nnz_num);
    vector<float> x(col_num,1.0);
    vector<float> y(row_num);
    vector<float> y_res(row_num);
    vector<float> y_cusparse_res(row_num);
    int iter = 2000;
    readMtxFile(is_weighted, row_num, nnz_num, row_offset.data(), col_index.data(), value.data(), file);

    // Print matrix basic information: rows, cols, nnz, mean/variance of nnz per row
    {
        long long nnz_ll = nnz_num;
        double mean = static_cast<double>(nnz_ll) / static_cast<double>(row_num);
        double var = 0.0;
        for (int i = 0; i < row_num; ++i) {
            double r = static_cast<double>(row_offset[i+1] - row_offset[i]);
            var += (r - mean) * (r - mean);
        }
        var = var / static_cast<double>(row_num);
        std::cout << "Matrix: rows=" << row_num << " cols=" << col_num << " nnz=" << nnz_num << std::endl;
        std::cout << "Mean nnz/row=" << mean << " Variance nnz/row=" << var << std::endl;
    }

    // check input
    // std::cout<<" The row_offset is: "<<std::endl;
    // vec_print<int>(row_offset);
    // std::cout<<" The col_index is: "<<std::endl;
    // vec_print<int>(col_index);
    // std::cout<<" The value is: "<<std::endl;
    // vec_print<float>(value);

    // allocate memory in GPU device
    int* d_A_row_offset;
    int* d_A_col_index;
    float* d_A_value;
    float* d_x;
    float* d_y;
    float* d_y_cusparse;

    checkCudaErrors(cudaMalloc(&d_A_row_offset, (row_num + 1)*sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_A_col_index, nnz_num*sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_A_value, nnz_num*sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_x, col_num*sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_y, row_num*sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_y_cusparse, row_num*sizeof(float)));
    checkCudaErrors(cudaMemcpy( d_A_row_offset, row_offset.data(), (row_num + 1)*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_A_col_index, col_index.data(), nnz_num*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_A_value, value.data(), nnz_num*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_x, x.data(), col_num*sizeof(float), cudaMemcpyHostToDevice));
    
    // spmv
    // Basic stats
    int mean_col_num = (nnz_num + (row_num - 1)) / row_num;
    std::cout << "The average col num is: " << mean_col_num << std::endl;

    // Build row-length buckets for better load balance
    // Buckets: [1], [2], [3-4], [5-8], [9-16], [17-32], [>32]
    vector<int> bucket_1;
    vector<int> bucket_2;
    vector<int> bucket_4;
    vector<int> bucket_8;
    vector<int> bucket_16;
    vector<int> bucket_32;
    vector<int> bucket_long;
    bucket_1.reserve(row_num);
    bucket_2.reserve(row_num);
    bucket_4.reserve(row_num);
    bucket_8.reserve(row_num);
    bucket_16.reserve(row_num);
    bucket_32.reserve(row_num);
    bucket_long.reserve(row_num);

    for (int r = 0; r < row_num; r++) {
        int row_nnz = row_offset[r + 1] - row_offset[r];
        if (row_nnz <= 1) bucket_1.push_back(r);
        else if (row_nnz == 2) bucket_2.push_back(r);
        else if (row_nnz <= 4) bucket_4.push_back(r);
        else if (row_nnz <= 8) bucket_8.push_back(r);
        else if (row_nnz <= 16) bucket_16.push_back(r);
        else if (row_nnz <= 32) bucket_32.push_back(r);
        else bucket_long.push_back(r);
    }
    std::cout << "Bucket sizes: "
              << "b1=" << bucket_1.size() << ", "
              << "b2=" << bucket_2.size() << ", "
              << "b4=" << bucket_4.size() << ", "
              << "b8=" << bucket_8.size() << ", "
              << "b16=" << bucket_16.size() << ", "
              << "b32=" << bucket_32.size() << ", "
              << "blong=" << bucket_long.size() << std::endl;

    auto alloc_and_copy_rows = [&](const vector<int>& host_rows) -> int* {
        if (host_rows.empty()) return nullptr;
        int* d_rows = nullptr;
        checkCudaErrors(cudaMalloc(&d_rows, host_rows.size() * sizeof(int)));
        checkCudaErrors(cudaMemcpy(d_rows, host_rows.data(), host_rows.size() * sizeof(int), cudaMemcpyHostToDevice));
        return d_rows;
    };

    int* d_bucket_1 = alloc_and_copy_rows(bucket_1);
    int* d_bucket_2 = alloc_and_copy_rows(bucket_2);
    int* d_bucket_4 = alloc_and_copy_rows(bucket_4);
    int* d_bucket_8 = alloc_and_copy_rows(bucket_8);
    int* d_bucket_16 = alloc_and_copy_rows(bucket_16);
    int* d_bucket_32 = alloc_and_copy_rows(bucket_32);
    int* d_bucket_long = alloc_and_copy_rows(bucket_long);
    // Run and time CPU reference implementation (single run)
    {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        spmv_cpu_kernel(row_offset, col_index, value, x, y_res, row_num);
        auto cpu_stop = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(cpu_stop - cpu_start).count();
        std::cout << "CPU SpMV time = " << cpu_ms << " ms" << std::endl;
    }

    // const int THREADS_PER_VECTOR = 32;
    // const unsigned int VECTORS_PER_BLOCK  = 256 / THREADS_PER_VECTOR;
    // const unsigned int THREADS_PER_BLOCK  = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;
    // const unsigned int NUM_BLOCKS = static_cast<unsigned int>((row_num + (VECTORS_PER_BLOCK - 1)) / VECTORS_PER_BLOCK);
    // My_spmv_csr_kernel<int, float, VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<NUM_BLOCKS, THREADS_PER_BLOCK>>> 
    //     (row_num, d_A_row_offset, d_A_col_index, d_A_value, d_x, d_y);

    // Time bucketed custom kernels over `iter` iterations using CUDA events
    cudaEvent_t kstart, kstop;
    checkCudaErrors(cudaEventCreate(&kstart));
    checkCudaErrors(cudaEventCreate(&kstop));
    checkCudaErrors(cudaEventRecord(kstart));
    for(int i=0; i<iter; i++){
        // For each bucket, keep 256 threads/block and adjust vectors-per-block.
        // THREADS_PER_VECTOR must be power-of-two and <= 32.
        if (!bucket_1.empty()) {
            const int THREADS_PER_VECTOR = 1;
            const unsigned int VECTORS_PER_BLOCK = 256 / THREADS_PER_VECTOR;
            const unsigned int NUM_BLOCKS = static_cast<unsigned int>((bucket_1.size() + (VECTORS_PER_BLOCK - 1)) / VECTORS_PER_BLOCK);
            My_spmv_csr_bucket_kernel<int, float, VECTORS_PER_BLOCK, THREADS_PER_VECTOR>
                <<<NUM_BLOCKS, 256>>>(static_cast<int>(bucket_1.size()), d_bucket_1, d_A_row_offset, d_A_col_index, d_A_value, d_x, d_y);
        }
        if (!bucket_2.empty()) {
            const int THREADS_PER_VECTOR = 2;
            const unsigned int VECTORS_PER_BLOCK = 256 / THREADS_PER_VECTOR;
            const unsigned int NUM_BLOCKS = static_cast<unsigned int>((bucket_2.size() + (VECTORS_PER_BLOCK - 1)) / VECTORS_PER_BLOCK);
            My_spmv_csr_bucket_kernel<int, float, VECTORS_PER_BLOCK, THREADS_PER_VECTOR>
                <<<NUM_BLOCKS, 256>>>(static_cast<int>(bucket_2.size()), d_bucket_2, d_A_row_offset, d_A_col_index, d_A_value, d_x, d_y);
        }
        if (!bucket_4.empty()) {
            const int THREADS_PER_VECTOR = 4;
            const unsigned int VECTORS_PER_BLOCK = 256 / THREADS_PER_VECTOR;
            const unsigned int NUM_BLOCKS = static_cast<unsigned int>((bucket_4.size() + (VECTORS_PER_BLOCK - 1)) / VECTORS_PER_BLOCK);
            My_spmv_csr_bucket_kernel<int, float, VECTORS_PER_BLOCK, THREADS_PER_VECTOR>
                <<<NUM_BLOCKS, 256>>>(static_cast<int>(bucket_4.size()), d_bucket_4, d_A_row_offset, d_A_col_index, d_A_value, d_x, d_y);
        }
        if (!bucket_8.empty()) {
            const int THREADS_PER_VECTOR = 8;
            const unsigned int VECTORS_PER_BLOCK = 256 / THREADS_PER_VECTOR;
            const unsigned int NUM_BLOCKS = static_cast<unsigned int>((bucket_8.size() + (VECTORS_PER_BLOCK - 1)) / VECTORS_PER_BLOCK);
            My_spmv_csr_bucket_kernel<int, float, VECTORS_PER_BLOCK, THREADS_PER_VECTOR>
                <<<NUM_BLOCKS, 256>>>(static_cast<int>(bucket_8.size()), d_bucket_8, d_A_row_offset, d_A_col_index, d_A_value, d_x, d_y);
        }
        if (!bucket_16.empty()) {
            const int THREADS_PER_VECTOR = 16;
            const unsigned int VECTORS_PER_BLOCK = 256 / THREADS_PER_VECTOR;
            const unsigned int NUM_BLOCKS = static_cast<unsigned int>((bucket_16.size() + (VECTORS_PER_BLOCK - 1)) / VECTORS_PER_BLOCK);
            My_spmv_csr_bucket_kernel<int, float, VECTORS_PER_BLOCK, THREADS_PER_VECTOR>
                <<<NUM_BLOCKS, 256>>>(static_cast<int>(bucket_16.size()), d_bucket_16, d_A_row_offset, d_A_col_index, d_A_value, d_x, d_y);
        }
        if (!bucket_32.empty()) {
            const int THREADS_PER_VECTOR = 32;
            const unsigned int VECTORS_PER_BLOCK = 256 / THREADS_PER_VECTOR;
            const unsigned int NUM_BLOCKS = static_cast<unsigned int>((bucket_32.size() + (VECTORS_PER_BLOCK - 1)) / VECTORS_PER_BLOCK);
            My_spmv_csr_bucket_kernel<int, float, VECTORS_PER_BLOCK, THREADS_PER_VECTOR>
                <<<NUM_BLOCKS, 256>>>(static_cast<int>(bucket_32.size()), d_bucket_32, d_A_row_offset, d_A_col_index, d_A_value, d_x, d_y);
        }
        if (!bucket_long.empty()) {
            const int BLOCK_SIZE = 256;
            const unsigned int NUM_BLOCKS = static_cast<unsigned int>(bucket_long.size());
            My_spmv_csr_block_per_row_kernel<BLOCK_SIZE>
                <<<NUM_BLOCKS, BLOCK_SIZE>>>(static_cast<int>(bucket_long.size()), d_bucket_long, d_A_row_offset, d_A_col_index, d_A_value, d_x, d_y);
        }
    }
    checkCudaErrors(cudaEventRecord(kstop));
    checkCudaErrors(cudaEventSynchronize(kstop));
    float kernel_ms = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&kernel_ms, kstart, kstop));
    std::cout << "Custom kernel total time (" << iter << " iters) = " << kernel_ms << " ms, avg = " << (kernel_ms / iter) << " ms/iter" << std::endl;
    checkCudaErrors(cudaEventDestroy(kstart));
    checkCudaErrors(cudaEventDestroy(kstop));

    checkCudaErrors(cudaMemcpy(y.data(), d_y, row_num*sizeof(float), cudaMemcpyDeviceToHost));

    // cusparse spmv
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    float     alpha           = 1.0f;
    float     beta            = 0.0f;

    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, row_num, col_num, nnz_num,
                                      d_A_row_offset, d_A_col_index, d_A_value,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense vector X
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, col_num, d_x, CUDA_R_32F) )
    // Create dense vector y
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, row_num, d_y_cusparse, CUDA_R_32F) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                 handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMV
    // Time cuSPARSE SpMV over `iter` iterations using CUDA events
    cudaEvent_t sstart, sstop;
    checkCudaErrors(cudaEventCreate(&sstart));
    checkCudaErrors(cudaEventCreate(&sstop));
    checkCudaErrors(cudaEventRecord(sstart));
    for(int i=0; i<iter; i++){
        CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer) )
    }
    checkCudaErrors(cudaEventRecord(sstop));
    checkCudaErrors(cudaEventSynchronize(sstop));
    float cusparse_ms = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&cusparse_ms, sstart, sstop));
    std::cout << "cuSPARSE total time (" << iter << " iters) = " << cusparse_ms << " ms, avg = " << (cusparse_ms / iter) << " ms/iter" << std::endl;
    checkCudaErrors(cudaEventDestroy(sstart));
    checkCudaErrors(cudaEventDestroy(sstop));

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(y_cusparse_res.data(), d_y_cusparse, row_num * sizeof(float),
                           cudaMemcpyDeviceToHost) )

    bool check_result = true;
    for(int i=0; i<row_num; i++){
        if(fabs(y[i]-y_cusparse_res[i])>1e-3){
            std::cout<<"The result is error!"<<std::endl;
            printf("The row is: %d the y is:%f and the cusparse_y is:%f\n", i, y[i], y_cusparse_res[i]);
            check_result = false;
            break;
        }
    }
    if(check_result){
        std::cout<<"The result is right!"<<std::endl;
    }

    // Free Memory
    if (d_bucket_1) cudaFree(d_bucket_1);
    if (d_bucket_2) cudaFree(d_bucket_2);
    if (d_bucket_4) cudaFree(d_bucket_4);
    if (d_bucket_8) cudaFree(d_bucket_8);
    if (d_bucket_16) cudaFree(d_bucket_16);
    if (d_bucket_32) cudaFree(d_bucket_32);
    if (d_bucket_long) cudaFree(d_bucket_long);
    cudaFree(d_A_row_offset);
    cudaFree(d_A_col_index);
    cudaFree(d_A_value);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}