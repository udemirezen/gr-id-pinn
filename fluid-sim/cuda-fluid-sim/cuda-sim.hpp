#ifndef CUDA_SIM_HPP
#define CUDA_SIM_HPP


// Binlienar interpolation algorithm shamelessly stolen / adapted from
// https://stackoverflow.com/questions/21128731/bilinear-interpolation-in-c-c-and-cuda
__global__ void bilinear_interpolation_kernel_GPU(float2 * __restrict__ d_result,
                                                  const float2 * __restrict__ d_data,
                                                  const float * __restrict__ d_xout,
                                                  const float * __restrict__ d_yout, 
                                                  const int M1, const int M2,
                                                  const int N1, const int N2)
{
   const int l = threadIdx.x + blockDim.x * blockIdx.x;
   const int k = threadIdx.y + blockDim.y * blockIdx.y;

    if ((l<N1)&&(k<N2)) {

        float2 result_temp1, result_temp2;

        const int    ind_x = floor(d_xout[k*N1+l]); 
        const float  a     = d_xout[k*N1+l]-ind_x; 

        const int    ind_y = floor(d_yout[k*N1+l]); 
        const float  b     = d_yout[k*N1+l]-ind_y; 

        float2 d00, d01, d10, d11;
        if (((ind_x)   < M1)&&((ind_y)   < M2))
        	d00 = d_data[ind_y*M1+ind_x];
        else
        	d00 = make_float2(0.f, 0.f);
        
        if (((ind_x+1) < M1)&&((ind_y)   < M2))
        	d10 = d_data[ind_y*M1+ind_x+1];
        else
        	d10 = make_float2(0.f, 0.f);
        
        if (((ind_x)   < M1)&&((ind_y+1) < M2))
        	d01 = d_data[(ind_y+1)*M1+ind_x];
        else
        	d01 = make_float2(0.f, 0.f);
        
        if (((ind_x+1) < M1)&&((ind_y+1) < M2))
        	d11 = d_data[(ind_y+1)*M1+ind_x+1];
        else
        	d11 = make_float2(0.f, 0.f);

        result_temp1.x = a * d10.x + (-d00.x * a + d00.x); 
        result_temp1.y = a * d10.y + (-d00.y * a + d00.y);

        result_temp2.x = a * d11.x + (-d01.x * a + d01.x);
        result_temp2.y = a * d11.y + (-d01.y * a + d01.y);

        d_result[k*N1+l].x = b * result_temp2.x + (-result_temp1.x * b + result_temp1.x);
        d_result[k*N1+l].y = b * result_temp2.y + (-result_temp1.y * b + result_temp1.y);

    } 
}

__device__
void advect(float2 * x_out,  // The advected quantity
            float timestep,  // what it sounds like 
            float rdx,  // Grid scale
            float2 * u,  // Velocity
            float2 * x)  // Quantity to advect
{
	
}

void diffuse() {

}

void addForces() {

float2

}

void computePressure() {

}

void subtractForces() {

}