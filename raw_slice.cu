#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <netdb.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include "gputimer.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include "helper_functions.h"
#include "helper_cuda.h"


/* FILTER PARAMETERS

        Low pass Direct-form FIR
        Order 262144
        Window Kaiser
        Fs = 61440000 Hz
        Fpass = 190497.31 Hz
        Fstop = 192000.0 Hz
        Apass = 0.1 db
        Astop = 100 db

*/

// comment out to speed up execution by 30% 
//#define USE_DBL_PRECISION_FFT

#define DFT_BLOCK_SIZE 1048576    // 1M real samples
#define L_SIZE 786432             // Number of new input samples consumed per data block 

#define COMPLEX_SIGNAL_SIZE (DFT_BLOCK_SIZE/2 + 1)   // R2C FFT length

#define P_SIZE 262145             // FIR Length
#define V_SIZE 4                  // Overlap factor  V = N/(P-1)
#define SAMPLING_RATE 61440000    // Hz
#define D_size 128                // decimation factor
#define LO 14040078.125           // LO
//#define N_ROT  240300       // LO = 14080078.125 Hz  Nrot = round(COMPLEX_SIGNAL_SIZE*LO/V_SIZE*SAMPLING_RATE) * V_SIZE
#define SAMPLE_LEN 16       // 16 bit real samples

// mix to baseband formula
// new_index = (index >= rot) ? index - N_ROT : FFT_SIZE - N_ROT + index

// Complex multiplication
static __device__ inline cufftComplex ComplexMul(cufftComplex a, cufftComplex b)
{
    cufftComplex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

// Complex scale
static __device__ inline cufftComplex ComplexScale(cufftComplex a, float s)
{
    cufftComplex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

static __device__ inline cufftComplex ComplexAdd(cufftComplex a, cufftComplex b)
{
    cufftComplex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

__global__
void gpu_process_buffer(cufftReal * d_signal, const short * d_buffer, cufftReal * d_delay_line)
{
    
    //  This is the overlap-save routine and conversion from short to float
    //  Not sure about scaling factor 256.0 

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // copy the last P-1 samples to the beginning ofthe window
    if (tid < P_SIZE - 1) {
        d_signal[tid] = d_delay_line[tid];

        // store the tail of the buffer in the delay line
        d_delay_line[tid] = d_buffer[L_SIZE - (P_SIZE -1) + tid ]/256.0f;
    } 
    else
    {
        // add buffer to the fft window
        d_signal[tid] = d_buffer[tid - (P_SIZE - 1)]/256.0f;
    }


}

__global__
void gpu_mix_and_convolve(const cufftComplex *d_fft, const cufftComplex *d_fir_fft, 
                                cufftComplex * d_receiver, cufftComplex * d_receiver2,
                                const int nrot, const float scale)
{
    const size_t numThreads = blockDim.x * gridDim.x;
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // mix to baseband

    size_t new_index;
    //size_t new_index = (tid >= nrot) ? tid - nrot : COMPLEX_SIGNAL_SIZE - nrot + tid;
    //d_receiver[tid] = d_fft[new_index];

    for (int i = tid; i < COMPLEX_SIGNAL_SIZE; i += numThreads)
    {
        new_index = (i >= nrot) ? i - nrot : COMPLEX_SIGNAL_SIZE - nrot + i;
        d_receiver[i]  = d_fft[new_index];
        d_receiver[i] = ComplexScale(ComplexMul(d_receiver[i], d_fir_fft[i]), scale);
        
    }

}

__global__
void gpu_decimate(const cufftComplex * d_receiver, cufftComplex * d_slice)
{
    //const int numThreads = blockDim.x * gridDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid > COMPLEX_SIGNAL_SIZE/D_size) return;

    cufftComplex decimated_bin;
    decimated_bin.x = 0;
    decimated_bin.y = 0;

    // fold FFT back on itself
    for (int i = tid; i < COMPLEX_SIGNAL_SIZE; i += COMPLEX_SIGNAL_SIZE/D_size) 
    {
        decimated_bin = ComplexAdd(decimated_bin, d_receiver[i]);
    }

    d_slice[tid] = decimated_bin;
    
}

int
main(int argc, char **argv)
{

   
    // create buffers
    size_t fft_result_size = sizeof(cufftComplex)*COMPLEX_SIGNAL_SIZE;
    size_t buffer_size = sizeof(short)*L_SIZE;
    size_t delay_line_size = sizeof(cufftReal) * (P_SIZE - 1);
    size_t d_signal_size = sizeof(cufftReal)*DFT_BLOCK_SIZE;

    size_t d_fir_size = sizeof(cufftReal)*DFT_BLOCK_SIZE;
    size_t d_fir_fft_size = sizeof(cufftComplex)*COMPLEX_SIGNAL_SIZE;

    size_t rx_td_size = sizeof(cufftReal)*DFT_BLOCK_SIZE/D_size;


    // findCudaDevice(argc, (const char **)argv);
    // Set flag to enable zero copy access
    cudaSetDeviceFlags(cudaDeviceMapHost);

    GpuTimer timer;
    timer.Start();


    // Allocate host-device mapped memory for the fir and buffer
    
    unsigned char *h_buffer = NULL;  // input stream buffer for ADC samples
    cufftReal *h_fir = NULL;   // host buffer for the FIR filter coefs
    cufftReal *h_rx_td = NULL; // decimated and filtered signal goes here

    cudaHostAlloc((void **)&h_fir, d_fir_size, cudaHostAllocMapped); 
    cudaHostAlloc((void **)&h_buffer, buffer_size, cudaHostAllocMapped); 
    cudaHostAlloc((void **)&h_rx_td, rx_td_size, cudaHostAllocMapped); 
    
    cufftReal *d_signal;  // time domain input signal for overlap-save
    cufftComplex *d_fft;  // DFT of the input signal

    cufftReal *d_delay_line;  // tail of each DFT window for overlap-save
    short *d_buffer; // device pointer to h_buffer
    cufftReal *d_fir;   // device pointer to h_fir
    cufftComplex *d_fir_fft;  // FFT of the FIR filter for fast convolution
    cufftReal *d_rx_td;  // device pointer to h_rx_td


    // get device pointers for the mapped buffers
    //cudaHostGetDevicePointer((void **)&d_signal, (void *) h_signal, 0);
    cudaHostGetDevicePointer((void **)&d_fir, (void *) h_fir, 0);
    cudaHostGetDevicePointer((void **)&d_buffer, (void *) h_buffer, 0);
    cudaHostGetDevicePointer((void **)&d_rx_td, (void *) h_rx_td, 0);

    // allocate device memory for overlap-save
    cudaMalloc((void **)&d_delay_line, delay_line_size);
    cudaMalloc((void **)&d_signal, d_signal_size);
    cudaMalloc((void **)&d_fft, fft_result_size);
    cudaMalloc((void **)&d_fir_fft, d_fir_fft_size);

  
    // zero out buffers using the GPU
    // we can speed this up by using cudeMemsetAsync 
    cudaMemset(d_signal, 0, d_signal_size);
    cudaMemset(d_buffer, 0, buffer_size);
    cudaMemset(d_delay_line, 0, delay_line_size);
    cudaMemset(d_fir, 0, d_fir_size);
    cudaMemset(d_rx_td, 0, rx_td_size);


    //allocate receiver
    cufftComplex *d_receiver = NULL;
    cufftComplex *d_slice = NULL;
    cudaMalloc((void **)&d_receiver, fft_result_size);
    cudaMalloc((void **)&d_slice, fft_result_size/D_size);

    cufftComplex *d_receiver2 = NULL;
    cudaMalloc((void **)&d_receiver2, fft_result_size);

    // ready to start processing input

    FILE *firfile;
    size_t samples_read=0;
    size_t total_samples_read=0;

    /* load filter coeffs */
    // fir file format one ASCII coeff per line
    char line[80];
    firfile=fopen("240khz.fir","r");
    //firfile=fopen("240khz-fir-float","r");
    //firfile=fopen("192khz-fir.fcf","r");

    

    /* did it open? */
    if (firfile == NULL)
    {
      fprintf(stderr, "ERROR opening filter file. aborting.\n");
      exit(1);
    }

    for (int i=0; i < P_SIZE; i++)
    {
        if (fgets(line, sizeof line, firfile) != NULL)
        {
            h_fir[i] = strtof(line, NULL);
            //printf("read coef: %.30f\n", h_fir[i]);
        }
        else
        {
          fprintf(stderr, "ERROR reading filter coefficients. aborting.\n");
          exit(1);
        }


    }

    // CUFFT plan
    //cufftHandle planZ;  //double precision plan
    //cufftPlan1d(&planZ, DFT_BLOCK_SIZE, CUFFT_D2Z, 1);


     #ifdef USE_DBL_PRECISION_FFT
            cufftHandle planZ2D; // double precision IFFT plan
            cufftPlan1d(&planZ2D, DFT_BLOCK_SIZE, CUFFT_Z2D, 1);
            
     #else
            cufftHandle planC;  //single precition plan
            cufftHandle planC2R; // single precision IFFT plan
            cufftPlan1d(&planC, DFT_BLOCK_SIZE, CUFFT_R2C, 1);
            cufftPlan1d(&planC2R, DFT_BLOCK_SIZE/D_size, CUFFT_C2R, 1);
            //cufftDestroy(planZ);
     #endif

    //calclulate and store FFT of the FIR filter
    cufftExecR2C(planC, d_fir, d_fir_fft);


    //cudaDeviceSynchronize(); getLastCudaError("Kernel execution failed [ FIR FFT ]");

    // calculate FFT bin rotation value for the mixer 
    double nrot = round((double)LO*(COMPLEX_SIGNAL_SIZE-1) / ((double)V_SIZE*SAMPLING_RATE)) * V_SIZE;
    fprintf(stderr, "FFT rotation %d bins resulting in LO %.11g Hz\n", (int)nrot, nrot*SAMPLING_RATE/COMPLEX_SIGNAL_SIZE);

    int discard_size = (P_SIZE-1)/D_size;
    int td_size = L_SIZE/D_size;
    fprintf(stderr, "IFFT discard samples %d, keep %d, output sample size %d bytes(floating pt)\n", discard_size, td_size, sizeof(cufftReal));

    timer.Stop();
    fprintf(stderr, "Setup complete in %g ms\n", timer.Elapsed());
    
    timer.Start();

    int skip = 2;  // number of initial frames to skip

    for(;;)
    {
        /* read from stdin until it's end */
        samples_read = fread(h_buffer, sizeof(short), L_SIZE, stdin);
        total_samples_read += samples_read;

        if (samples_read < buffer_size) 
        {
                if (feof(stdin)) break;
        }
        
        //fprintf(stdout, "read %d samples\n", samples_read);
        gpu_process_buffer<<<DFT_BLOCK_SIZE/1024, 1024>>>(d_signal, d_buffer, d_delay_line);
    
        // Check if kernel execution generated and error - this slows down execution
        // cudaDeviceSynchronize(); getLastCudaError("Kernel execution failed [ gpu_process_buffer ]");

        #ifdef USE_DBL_PRECISION_FFT
                //call double precision real-to-complex FFT
                cufftExecD2Z(planZ, d_signal, d_fft);
        #else 
                //call single precision real-to-complex FFT
                cufftExecR2C(planC, (cufftReal *) d_signal, (cufftComplex *) d_fft);
        #endif

        gpu_mix_and_convolve<<<128, 1024>>>(d_fft, d_fir_fft, d_receiver, d_receiver2, (int)nrot, 1.0f/COMPLEX_SIGNAL_SIZE);

        gpu_decimate<<<COMPLEX_SIGNAL_SIZE/(1024*D_size), 1024>>>(d_receiver,d_slice);

        #ifdef USE_DBL_PRECISION_FFT
                cufftExecZ2D(planZ2D, d_slice, d_rx_td);
        #else
                cufftExecC2R(planC2R, (cufftComplex *) d_slice, (cufftReal *) d_rx_td);
        #endif

        //cudaDeviceSynchronize(); getLastCudaError("Kernel execution failed [ IFFT ]");

        if (skip == 0) 
        {
            fwrite(h_rx_td+discard_size, sizeof(cufftReal), td_size, stdout);
        }
        else
        {
            skip--;
        }
        
        
    }

    timer.Stop();
    float secs = (float)total_samples_read / SAMPLING_RATE;
    fprintf(stderr, "Processed %d samples (%g sec) signal in %g ms, performance ratio: %g\n", total_samples_read, secs, timer.Elapsed(), secs*1000/timer.Elapsed());

    #ifdef USE_DBL_PRECISION_FFT
            cufftDestroy(planZ);
            cufftDestroy(planZ2D);
    #else
            cufftDestroy(planC);
            cufftDestroy(planC2R);
    #endif

    cudaFree(d_fft);
    cudaFree(d_signal);
    cudaFree(d_slice);
    cudaFree(d_fir);
    cudaFree(d_fir_fft);
    cudaFree(d_buffer);
    cudaFree(d_delay_line);
}