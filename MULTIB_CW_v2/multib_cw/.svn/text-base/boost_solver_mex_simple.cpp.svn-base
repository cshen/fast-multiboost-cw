

//compile: mex cdboost_solver_mex_simple.cpp -lmwblas -largeArrayDims


#include <cmath>
#include "mex.h"
#include "matrix.h"
#include "blas.h"
#include <time.h>
#include <limits>
#include <cstdlib>
#include <vector>
#include <iostream>
#include "cp_matlib.h"

using namespace std;


const mxArray* init_w; 
const mxArray* init_exp_m_wh;
const mxArray* update_dim_idxes;

char* p_sel_mat;
char* n_sel_mat;

double tradeoff_nv; 

mxArray* new_w;
mxArray* exp_m_wh;


double ZERO_EPS(1e-14);


//===============================================================================================================



void do_cd_mex(){
       
    
    exp_m_wh=mxDuplicateArray(init_exp_m_wh);
    size_t pair_num=mxGetM(exp_m_wh);
    
    new_w=mxDuplicateArray(init_w);
    double * new_w_ptr=mxGetPr(new_w);
    
    double * exp_m_wh_ptr=mxGetPr(exp_m_wh);
    double * update_dim_idxes_ptr=mxGetPr(update_dim_idxes);
     
    char * one_p_sels_ptr=new char[pair_num];
    char * one_n_sels_ptr=new char[pair_num];
              
    size_t cd_iter_num=mxGetM(update_dim_idxes);
            
        
    double tmp_v1(tradeoff_nv/2.0);
    double tmp_v2(tmp_v1*tmp_v1);
    double feat_v(1.0);
    double minus_feat_v(-1.0);
    
   for (size_t cd_iter_idx(0); cd_iter_idx<cd_iter_num; ++cd_iter_idx){
        
        size_t update_dim_idx=(size_t)update_dim_idxes_ptr[cd_iter_idx];
    
        get_mat_col_char(p_sel_mat, update_dim_idx, one_p_sels_ptr, pair_num);
        get_mat_col_char(n_sel_mat, update_dim_idx, one_n_sels_ptr, pair_num);
                
        double one_old_dim=new_w_ptr[update_dim_idx];
                
        double sel_bar_W_ps(0);
        double sel_bar_W_ns(0);
                
        double tmp_pv(exp(one_old_dim));
        double tmp_nv(exp(-1.0*one_old_dim));
        
        
        for (size_t i(0); i<pair_num; ++i){
            if (one_p_sels_ptr[i]){
                sel_bar_W_ps+=exp_m_wh_ptr[i]*tmp_pv;
            }else if (one_n_sels_ptr[i]){
                sel_bar_W_ns+=exp_m_wh_ptr[i]*tmp_nv;
            }
        }
        
              
        
        
        double one_new_dim(0);
        if (sel_bar_W_ps > sel_bar_W_ns){
            if (sel_bar_W_ns>ZERO_EPS){
                one_new_dim=log(sqrt(sel_bar_W_ps*sel_bar_W_ns + tmp_v2) - tmp_v1)- log(sel_bar_W_ns); 
            }else{
                one_new_dim=log(sel_bar_W_ps)-log(tradeoff_nv);
            }
        }
               
        
        double minus_delta_w_dims=one_old_dim-one_new_dim;
        
        
        if (abs(minus_delta_w_dims)>ZERO_EPS){
        
            new_w_ptr[update_dim_idx]=one_new_dim;
            
            double tmp_pv2(exp(minus_delta_w_dims));
            double tmp_nv2(exp(-1.0*minus_delta_w_dims));

            for (size_t i(0); i<pair_num; ++i){
                if (one_p_sels_ptr[i]){
                    exp_m_wh_ptr[i]=exp_m_wh_ptr[i]*tmp_pv2;
                }else if (one_n_sels_ptr[i]){
                    exp_m_wh_ptr[i]=exp_m_wh_ptr[i]*tmp_nv2;
                }
            }
        }
                   
    }
    
    
    delete[] one_p_sels_ptr;
    delete[] one_n_sels_ptr;
  
}   


//===============================================================================================================



void mexFunction( int nlhs, mxArray *plhs[], 
		  int nrhs, const mxArray*prhs[] ){ 
    
        
    if (nlhs < 2) {
       mexErrMsgTxt(" the num of output args is not correct!");
    }
    
    if (nrhs < 6) {
       mexErrMsgTxt(" the num of input args is not correct!");
    }
    
         
     tradeoff_nv=mxGetScalar(prhs[0]);
     update_dim_idxes=mxDuplicateArray(prhs[1]);
     init_w=prhs[2];
     p_sel_mat=(char*)mxGetData(prhs[3]);
     n_sel_mat=(char*)mxGetData(prhs[4]);
     init_exp_m_wh=prhs[5];
     
     // handle c++ and matlab start idx problem
     scalar_plus_mat(update_dim_idxes,-1);
         
     
     do_cd_mex();
     
     
     plhs[0]=new_w;
     plhs[1]=exp_m_wh;
     
}



