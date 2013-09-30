

#ifndef CP_MATLIB_H_
#define CP_MATLIB_H_


#include "matrix.h"
#include "blas.h"
#include <vector>
#include <assert.h>

using namespace std;

class Simple_mat{
public:
    double * ptr;
    size_t m;
    size_t n;
    bool sparse;
    
        
    Simple_mat():ptr(0),m(0), n(0), sparse(false){}
    
        
    inline virtual void create(const size_t &m1, const size_t &n1){
        m=m1;
        n=n1;
        //use () to init to zeros!!
        ptr=new double[m1*n1]();
        sparse=false;
    }
    
    Simple_mat(const size_t &m1, const size_t &n1){
        create(m1, n1);
    }
    
    inline virtual void destroy(){
        delete [] ptr;
    }
    
    inline virtual size_t get_e_num(){
        return m*n;
    }
    
    inline virtual void set_v(const size_t &m_idx, const size_t &n_idx, const double &v){
        ptr[n_idx*m+m_idx]=v;
    }
        
    inline virtual double get_v(const size_t &m_idx, const size_t &n_idx){
        return ptr[n_idx*m+m_idx];
    }
    
    inline virtual double clear(){
        size_t e_num=m*n;
        for (size_t i=0; i<e_num; ++i){
            ptr[i]=0;
        }
    }
    
};

class mxSimple_mat:public Simple_mat{
public:
    const mxArray * mx_mat;
    mwIndex *ir;
    mwIndex *jc;          
    
    virtual void init(const mxArray * mxA){
        assert(mxIsDouble(mxA));
        m=mxGetM(mxA);
        n=mxGetN(mxA);
        ptr=mxGetPr(mxA);
        mx_mat=mxA;
        
        if (mxIsSparse(mxA)){
            ir=mxGetIr(mxA);
            jc=mxGetJc(mxA);
            sparse=true;
        }else{
            ir=0;
            jc=0;
            sparse=false;
        }
    }
            
    //??? how to init parent
    mxSimple_mat():mx_mat(0), ir(0), jc(0){}
    
    mxSimple_mat(const mxArray * mxA){
        init(mxA);
    }
    
    inline virtual void create(const size_t &m1, const size_t &n1){
    }
    
    inline virtual void destroy(){
    }
    
};




class Mat_vec{
public:
    Simple_mat **ptr;
    size_t m;
    
    inline virtual void clear(){
        for (size_t i(0); i<m; ++i){ptr[i]=0;}
    }
    
    inline virtual void create(const size_t & m1){
        ptr=new Simple_mat*[m1];
        m=m1;
        clear();
    }
            
    Mat_vec(const size_t & m1){
        create(m1);
    }
    
    Mat_vec():ptr(0),m(0){}
            
    inline virtual void destroy(){
        for (size_t i(0); i<m; ++i){
            if (ptr[i]){
                ptr[i]->destroy();
            }
        }
        delete [] ptr;
    }
    
    inline virtual void set_mat(size_t i, Simple_mat * mat){
        ptr[i]=mat;
    }
    
    inline virtual Simple_mat * get_mat(size_t i){
        return ptr[i];
    }
    
};



class mxMat_vec:public Mat_vec{
public:
  
    inline virtual void create(const mxArray * cell_mat){
        assert(mxIsCell(cell_mat));
        m=mxGetNumberOfElements(cell_mat);
        Mat_vec::create(m);
        for (size_t i(0); i<m; ++i){
            mxArray * one_mat=mxGetCell(cell_mat, i);
            if (one_mat){
                ptr[i]=new mxSimple_mat(one_mat);
            }
        }
    }
        
    mxMat_vec(const mxArray * cell_mat){
        create(cell_mat);
    }
    
};


class Mat_vec_vec{
public:
    Mat_vec **ptr;
    size_t m;
    
    inline virtual void clear(){
        for (size_t i(0); i<m; ++i){ptr[i]=0;}
    }
    
    inline virtual void create(const size_t & m1){
        ptr=new Mat_vec*[m1];
        m=m1;
        clear();
    }
    
    Mat_vec_vec(const size_t & m1){
        create(m1);
    }
            
    Mat_vec_vec():ptr(0),m(0){}
            
    inline virtual void destroy(){
        for (size_t i(0); i<m; ++i){
            if (ptr[i]){
                ptr[i]->destroy();
            }
        }
        delete [] ptr;
    }
    
    inline virtual void set_mat_vec(size_t i, Mat_vec * mat){
        ptr[i]=mat;
    }
    
    inline virtual Mat_vec * get_mat_vec(size_t i){
        return ptr[i];
    }
};

class mxMat_vec_vec:public Mat_vec_vec{
public:
        
    inline virtual void create(const mxArray * cell_cell_mat){
        assert(mxIsCell(cell_cell_mat));
        m=mxGetNumberOfElements(cell_cell_mat);
        Mat_vec_vec::create(m);
        for (size_t i(0); i<m; ++i){
            mxArray *one_child_cell=mxGetCell(cell_cell_mat, i);
            ptr[i]=new mxMat_vec(one_child_cell);
        }
    }
        
    // ??? why need to repeat the parent's
    mxMat_vec_vec(const mxArray * cell_cell_mat){
        create(cell_cell_mat);
    }
    
    mxMat_vec_vec(const size_t & m1){
        Mat_vec_vec::create(m1);
    }
  
};

inline void des_mat(Simple_mat * v){
    v->destroy();
    delete v;
}

inline void des_mat_vec(Mat_vec * v){
    v->destroy();
    delete v;
}

inline void des_mat_vec_vec(Mat_vec_vec * v){
    v->destroy();
    delete v;
}


inline mxArray* copy_mat2mx(Simple_mat* mxB){
    size_t m=mxB->m;
    size_t n=mxB->n;
    mxArray *mxA =mxCreateDoubleMatrix(m,n, mxREAL);
    double *mxA_ptr=mxGetPr(mxA);
    double *mxB_ptr=mxB->ptr;
    size_t e_num = m*n;  
    for(size_t i(0);i<e_num;++i){
        mxA_ptr[i]=mxB_ptr[i];
    }
    return mxA;
}

inline mxArray * create_cell_mat(Mat_vec * mat_vec){
    size_t m=mat_vec->m;
    mxArray *mxA=mxCreateCellMatrix(m,1);
    Simple_mat ** ptr=mat_vec->ptr;
    for (size_t i(0); i<m; ++i){
        if (ptr[i]){
            mxSetCell(mxA,i, copy_mat2mx(ptr[i]));
        }
    }
    return mxA;
}


 inline mxArray * create_cell_mat(Mat_vec_vec * mat_vec_vec){
     size_t m=mat_vec_vec->m;
    mxArray *mxA=mxCreateCellMatrix(m,1);
    Mat_vec ** ptr=mat_vec_vec->ptr;
    for (size_t i(0); i<m; ++i){
        if (ptr[i]){
            mxArray *one_cell_mat=create_cell_mat(ptr[i]);
            mxSetCell(mxA,i, one_cell_mat);
        }
    }
    return mxA;
}


inline void dense_mat_x_mat(const mxArray* mxA, const mxArray* mxB, mxArray* mxC, bool At){
    
    char cht_tmp('T');
    char chn_tmp('N');
    
    
    double *A, *B, *C; /* pointers to input & output matrices*/
    ptrdiff_t m,n,m2,n2;      /* matrix dimensions */
    /* form of op(A) & op(B) to use in matrix multiplication */
    char *t_flag_A(0);
    char *t_flag_B(0);
    if(At){
        t_flag_A = &cht_tmp;
    }else{
        t_flag_A = &chn_tmp;
    }
    
    t_flag_B = &chn_tmp;
    
    /* scalar values to use in dgemm */
    double one = 1.0, zero = 0.0;

    A = mxGetPr(mxA); /* first input matrix */
    B = mxGetPr(mxB); /* second input matrix */
    /* dimensions of input matrices */
    m = mxGetM(mxA);  
    n = mxGetN(mxA);
    m2 = mxGetM(mxB);
    n2 = mxGetN(mxB);
    
    
    C = mxGetPr(mxC);

    if (!At) {
        if (n != m2) {
            mexErrMsgIdAndTxt("MATLAB:matrixMultiply:matchdims",
                "Inner dimensions of matrix multiply do not match.");
        }else{
            /* Pass arguments to Fortran by reference */
            dgemm(t_flag_A, t_flag_B, &m, &n2, &n, &one, A, &m, B, &m2, &zero, C, &m);
        }
    }
    
    if (At) {
        if (m != m2) {
            mexErrMsgIdAndTxt("MATLAB:matrixMultiply:matchdims",
                "Inner dimensions of matrix multiply do not match.");
        }else{
            /* Pass arguments to Fortran by reference */
            dgemm(t_flag_A, t_flag_B, &n, &n2, &m, &one, A, &m, B, &m2, &zero, C, &n);
        }
    }
}


      
inline void do_mat_x_mat(Simple_mat *mxA, Simple_mat* mxB, Simple_mat *mxC, bool At){
    
    char cht_tmp('T');
    char chn_tmp('N');
    
    
    double *A, *B, *C; /* pointers to input & output matrices*/
    ptrdiff_t m,n,m2,n2;      /* matrix dimensions */
    /* form of op(A) & op(B) to use in matrix multiplication */
    char *t_flag_A(0);
    char *t_flag_B(0);
    
    if(At){
        t_flag_A = &cht_tmp;
    }else{
        t_flag_A = &chn_tmp;
    }
    
    t_flag_B = &chn_tmp;
    
    /* scalar values to use in dgemm */
    double one = 1.0, zero = 0.0;

    A = mxA->ptr; /* first input matrix */
    B = mxB->ptr; /* second input matrix */
    /* dimensions of input matrices */
    m = mxA->m;  
    n = mxA->n;
    m2 = mxB->m;
    n2 = mxB->n;
        
    C = mxC->ptr;

    if (n != m2) {
        mexErrMsgIdAndTxt("MATLAB:matrixMultiply:matchdims",
            "Inner dimensions of matrix multiply do not match.");
    }else{
        /* Pass arguments to Fortran by reference */
        dgemm(t_flag_A, t_flag_B, &m, &n2, &n, &one, A, &m, B, &m2, &zero, C, &m);
    }
}


// mxArray* sparse_mat_x_mat(const mxArray* mxA, const mxArray* mxB){
//     int at(0), ac(0), bt(0), bc(0), ct(0), cc(0);
//     mxArray* mxC=ssmult (mxA, mxB, at, ac, bt, bc, ct, cc);
//     return mxC;
// }


inline mxArray * create_scalar_mat(const double & s){
    mxArray *smat = mxCreateDoubleMatrix(1,1,mxREAL);
    mxGetPr(smat)[0]=s;
    return smat;
}

inline size_t max_mat_idx(const Simple_mat &mxA){
    size_t e_num=mxA.m*mxA.n;
    size_t max_idx(0);
    double max_v(mxA.ptr[0]);
    for (size_t i(0); i<e_num; ++i){
        if(mxA.ptr[i]>max_v){
            max_v=mxA.ptr[i];
            max_idx=i;
        }
    }
    return max_idx;
}

inline size_t max_mat_idx(const mxArray *mxA){
    size_t e_num=mxGetNumberOfElements(mxA);
    size_t max_idx(0);
    double *mxA_ptr=mxGetPr(mxA);
    double max_v(mxA_ptr[0]);
    for (size_t i(0); i<e_num; ++i){
        if(mxA_ptr[i]>max_v){
            max_v=mxA_ptr[i];
            max_idx=i;
        }
    }
    return max_idx;
}

inline double max_mat_v(const mxArray *mxA){
    size_t e_num=mxGetNumberOfElements(mxA);
    double *mxA_ptr=mxGetPr(mxA);
    double max_v(mxA_ptr[0]);
    
    for (size_t i(0); i<e_num; ++i){
        if(mxA_ptr[i]>max_v){
            max_v=mxA_ptr[i];
        }
    }
    return max_v;
}

inline void mat_x_mat(const mxArray* mxA, const mxArray* mxB, mxArray* mxC){
    dense_mat_x_mat(mxA, mxB, mxC, false);
}

inline void mat_t_x_mat(const mxArray* mxA, const mxArray* mxB, mxArray* mxC){
    dense_mat_x_mat(mxA, mxB, mxC, true);
}

inline void mat_x_mat(Simple_mat *mxA, Simple_mat* mxB, Simple_mat *mxC){
    do_mat_x_mat(mxA, mxB, mxC, false);
}

inline void mat_t_x_mat(Simple_mat *mxA, Simple_mat* mxB, Simple_mat *mxC){
    do_mat_x_mat(mxA, mxB, mxC, true);
}


inline void mat_dotprod_mat(const mxArray* mxA, const mxArray* mxB, mxArray* mxC){
    double *mxA_ptr=mxGetPr(mxA);
    double *mxB_ptr=mxGetPr(mxB);
    size_t e_num=mxGetNumberOfElements(mxA);
    double *mxC_ptr=mxGetPr(mxC);
    for(size_t i(0);i<e_num;++i){
        mxC_ptr[i]= mxA_ptr[i] * mxB_ptr[i];
    }
}

inline void mat_dotprod_mat(Simple_mat* mxA, Simple_mat * mxB, Simple_mat* mxC){
    double *mxA_ptr=mxA->ptr;
    double *mxB_ptr=mxB->ptr;
    size_t e_num=mxA->get_e_num();
    double *mxC_ptr=mxC->ptr;
    for(size_t i(0);i<e_num;++i){
        mxC_ptr[i]= mxA_ptr[i] * mxB_ptr[i];
    }
}


inline void mat_x_mat_selcol_sparse(mxSimple_mat* mxA, mxSimple_mat* mxB, const size_t &col_idxB, Simple_mat *mxC){
    
    if(!mxA->sparse||!mxB->sparse){
        throw 100;
    }
            
    double *mxA_ptr=mxA->ptr;
    double *mxB_ptr=mxB->ptr;
    mwIndex *irA=mxA->ir;
    mwIndex *jcA=mxA->jc;
    mwIndex *irB=mxB->ir;
    mwIndex *jcB=mxB->jc;
       
    int one_col_first_e_idxB=jcB[col_idxB];
    int one_col_last_e_idxB=jcB[col_idxB+1]-1;
            
    mxC->clear();
    
    if (one_col_first_e_idxB>one_col_last_e_idxB){
        return;
    }
    
        
    double one_eA(0);
    size_t row_idxA(0);
    size_t col_numA=mxA->n;
    int one_col_first_e_idxA(0);
    int one_col_last_e_idxA(0);
    size_t row_idxB(0);
    
        
    for (size_t col_idxA(0); col_idxA<col_numA; ++col_idxA){
                
        one_col_first_e_idxA=jcA[col_idxA];
        one_col_last_e_idxA=jcA[col_idxA+1]-1;
               
        
        if (one_col_first_e_idxA>one_col_last_e_idxA){
            continue;
        }
                
    
        for (int e_idxA=one_col_first_e_idxA; e_idxA<=one_col_last_e_idxA; ++e_idxA){
            
                        
            row_idxA=irA[e_idxA];
            one_eA=mxA_ptr[e_idxA];
                        
            
            for (int e_idxB=one_col_first_e_idxB; e_idxB<=one_col_last_e_idxB; ++e_idxB){
              
                row_idxB=irB[e_idxB];
                if (col_idxA==row_idxB){
                    mxC->ptr[row_idxA]+=mxB_ptr[e_idxB]*one_eA;
                }
            }
        }
    }
    
}



inline void get_mat_col_sparse(mxSimple_mat* mxB, const size_t &col_idxB, Simple_mat *mxC){
    
    if(!mxB->sparse){
        throw 100;
    }
            
    double *mxB_ptr=mxB->ptr;
    mwIndex *irB=mxB->ir;
    mwIndex *jcB=mxB->jc;
       
    int one_col_first_e_idxB=jcB[col_idxB];
    int one_col_last_e_idxB=jcB[col_idxB+1]-1;
            
    mxC->clear();
    
    if (one_col_first_e_idxB>one_col_last_e_idxB){
        return;
    }
    
    size_t row_idxB(0);
        
    for (int e_idxB=one_col_first_e_idxB; e_idxB<=one_col_last_e_idxB; ++e_idxB){
        row_idxB=irB[e_idxB];
        mxC->ptr[row_idxB]=mxB_ptr[e_idxB];
    }
    
}




inline double iner_prod(const Simple_mat &mxA, const Simple_mat &mxB){
    size_t e_num=mxA.m*mxA.n;
    double *mxA_ptr=mxA.ptr;
    double *mxB_ptr=mxB.ptr;
    double ret(0);
    for(size_t i(0);i<e_num;++i){
        ret+= mxA_ptr[i] * mxB_ptr[i];
    }
    
    return ret;
}

inline void mat_minus_mat(const mxArray* mxA, const mxArray* mxB, mxArray* mxC){
    double *mxA_ptr=mxGetPr(mxA);
    double *mxB_ptr=mxGetPr(mxB);
    size_t e_num=mxGetNumberOfElements(mxA);
    double *mxC_ptr=mxGetPr(mxC);
    for(size_t i(0);i<e_num;++i){
        mxC_ptr[i]= mxA_ptr[i] - mxB_ptr[i];
    }
}

inline void mat_minus_mat(Simple_mat &mxA, const Simple_mat &mxB, Simple_mat &mxC){
    double *mxA_ptr=mxA.ptr;
    double *mxB_ptr=mxB.ptr;
    size_t e_num=mxA.get_e_num();
    double *mxC_ptr=mxC.ptr;
    for(size_t i(0);i<e_num;++i){
        mxC_ptr[i]= mxA_ptr[i] - mxB_ptr[i];
    }
}

inline void mat_plus_mat(const mxArray* mxA, const mxArray* mxB, mxArray* mxC){
    double *mxA_ptr=mxGetPr(mxA);
    double *mxB_ptr=mxGetPr(mxB);
    size_t e_num=mxGetNumberOfElements(mxA);
    double *mxC_ptr=mxGetPr(mxC);
    for(size_t i(0);i<e_num;++i){
        mxC_ptr[i]= mxA_ptr[i] + mxB_ptr[i];
    }
}


inline void mat_plus_mat(Simple_mat &mxA, Simple_mat &mxB, Simple_mat &mxC){
    double *mxA_ptr=mxA.ptr;
    double *mxB_ptr=mxB.ptr;
    size_t e_num=mxA.get_e_num();
    double *mxC_ptr=mxC.ptr;
    for(size_t i(0);i<e_num;++i){
        mxC_ptr[i]= mxA_ptr[i] + mxB_ptr[i];
    }
}


inline void mat_plus_mat(Simple_mat *mxA, Simple_mat *mxB, Simple_mat *mxC){
    double *mxA_ptr=mxA->ptr;
    double *mxB_ptr=mxB->ptr;
    size_t e_num=mxA->get_e_num();
    double *mxC_ptr=mxC->ptr;
    for(size_t i(0);i<e_num;++i){
        mxC_ptr[i]= mxA_ptr[i] + mxB_ptr[i];
    }
}

inline void mat_set_min(const mxArray* mxA, double scalar){
    double *mxA_ptr=mxGetPr(mxA);
    size_t e_num = mxGetNumberOfElements(mxA);  
    for(size_t i(0);i<e_num;++i){
        if (mxA_ptr[i]<scalar){
            mxA_ptr[i]=scalar;
        }
    }
}


inline void mat_set_min(Simple_mat* mxA, double scalar){
    double *mxA_ptr=mxA->ptr;
    size_t e_num = mxA->get_e_num();  
    for(size_t i(0);i<e_num;++i){
        if (mxA_ptr[i]<scalar){
            mxA_ptr[i]=scalar;
        }
    }
}

inline void scalar_x_mat(const mxArray* mxA, double scalar){
    double *mxA_ptr=mxGetPr(mxA);
    size_t e_num = mxGetNumberOfElements(mxA);  
    for(size_t i(0);i<e_num;++i){
        mxA_ptr[i]= mxA_ptr[i] * scalar;
    }
}

inline void scalar_x_mat(Simple_mat* mxA, double scalar){
    double *mxA_ptr=mxA->ptr;
    size_t e_num = mxA->get_e_num();  
    for(size_t i(0);i<e_num;++i){
        mxA_ptr[i]= mxA_ptr[i] * scalar;
    }
}

inline void scalar_x_mat(const mxArray* mxA, double scalar, const mxArray* mxC){
    double *mxA_ptr=mxGetPr(mxA);
    double *mxC_ptr=mxGetPr(mxC);
    size_t e_num = mxGetNumberOfElements(mxA);  
    for(size_t i(0);i<e_num;++i){
        mxC_ptr[i]= mxA_ptr[i] * scalar;
    }
}

inline void scalar_minus_mat(const mxArray* mxA, double scalar, const mxArray* mxC){
    double *mxA_ptr=mxGetPr(mxA);
    double *mxC_ptr=mxGetPr(mxC);
    size_t e_num = mxGetNumberOfElements(mxA);  
    for(size_t i(0);i<e_num;++i){
        mxC_ptr[i]= scalar - mxA_ptr[i];
    } 
}

inline void mat_minus_scalar(const Simple_mat &mxA, double scalar, Simple_mat &mxC){
    double *mxA_ptr=mxA.ptr;
    double *mxC_ptr=mxC.ptr;
    size_t e_num = mxA.m*mxA.n;  
    for(size_t i(0);i<e_num;++i){
        mxC_ptr[i]= mxA_ptr[i] - scalar;
    }
}

inline void scalar_minus_mat(const Simple_mat &mxA, double scalar, Simple_mat &mxC){
    double *mxA_ptr=mxA.ptr;
    double *mxC_ptr=mxC.ptr;
    size_t e_num = mxA.m*mxA.n;  
    for(size_t i(0);i<e_num;++i){
        mxC_ptr[i]= scalar - mxA_ptr[i];
    }
}


inline void scalar_plus_mat(const mxArray* mxA, double scalar){
    double *mxA_ptr=mxGetPr(mxA);
    size_t e_num = mxGetNumberOfElements(mxA);  
    for(size_t i(0);i<e_num;++i){
        mxA_ptr[i]= mxA_ptr[i] + scalar;
    }
}

inline void mat_plus_scalar(const mxArray* mxA, double scalar, const mxArray* mxC){
    double *mxA_ptr=mxGetPr(mxA);
    double *mxC_ptr=mxGetPr(mxC);
    size_t e_num = mxGetNumberOfElements(mxA);  
    for(size_t i(0);i<e_num;++i){
        mxC_ptr[i]= mxA_ptr[i] + scalar;
    }
}

inline void exp_mat(const mxArray* mxA){
    double *mxA_ptr=mxGetPr(mxA);
    size_t e_num = mxGetNumberOfElements(mxA);  
    for(size_t i(0);i<e_num;++i){
        mxA_ptr[i]= exp(mxA_ptr[i]);
    }
}

inline void exp_mat(Simple_mat * mxA){
    double *mxA_ptr=mxA->ptr;
    size_t e_num = mxA->get_e_num();  
    for(size_t i(0);i<e_num;++i){
        mxA_ptr[i]= exp(mxA_ptr[i]);
    }
}

inline void log_mat(Simple_mat * mxA){
    double *mxA_ptr=mxA->ptr;
    size_t e_num = mxA->get_e_num();  
    for(size_t i(0);i<e_num;++i){
        mxA_ptr[i]= log(mxA_ptr[i]);
    }
}

inline void sqrt_mat(const mxArray* mxA){
    double *mxA_ptr=mxGetPr(mxA);
    size_t e_num = mxGetNumberOfElements(mxA);  
    for(size_t i(0);i<e_num;++i){
        mxA_ptr[i]= sqrt(mxA_ptr[i]);
    }
}

inline void sqrt_mat(Simple_mat * mxA){
    double *mxA_ptr=mxA->ptr;
    size_t e_num = mxA->get_e_num();  
    for(size_t i(0);i<e_num;++i){
        mxA_ptr[i]= sqrt(mxA_ptr[i]);
    }
}

inline void square_mat(Simple_mat * mxA){
    double *mxA_ptr=mxA->ptr;
    size_t e_num = mxA->get_e_num();  
    for(size_t i(0);i<e_num;++i){
        mxA_ptr[i]= mxA_ptr[i]*mxA_ptr[i];
    }
}

inline void log_mat(const mxArray* mxA){
    double *mxA_ptr=mxGetPr(mxA);
    size_t e_num = mxGetNumberOfElements(mxA);  
    for(size_t i(0);i<e_num;++i){
        mxA_ptr[i]= log(mxA_ptr[i]);
    }
}

inline double sum_mat(const mxArray* mxA){
    double sum(0);
    double *mxA_ptr=mxGetPr(mxA);
    size_t e_num = mxGetNumberOfElements(mxA);  
    for(size_t i(0);i<e_num;++i){
        sum+= mxA_ptr[i];
    }
    return sum;
}

inline void copy_mat_fromB2A(const mxArray* mxA, const mxArray* mxB){
    double *mxA_ptr=mxGetPr(mxA);
    double *mxB_ptr=mxGetPr(mxB);
    size_t e_num = mxGetNumberOfElements(mxA);  
    for(size_t i(0);i<e_num;++i){
        mxA_ptr[i]=mxB_ptr[i];
    }
}

inline mxArray* copy_mat(const mxArray* mxB){
    size_t m=mxGetM(mxB);
    size_t n=mxGetN(mxB);
    mxArray *mxA =mxCreateDoubleMatrix(m,n, mxREAL);
    double *mxA_ptr=mxGetPr(mxA);
    double *mxB_ptr=mxGetPr(mxB);
    size_t e_num = m*n;  
    for(size_t i(0);i<e_num;++i){
        mxA_ptr[i]=mxB_ptr[i];
    }
    return mxA;
}

inline Simple_mat* copy_mat(Simple_mat* mxB){
    size_t m=mxB->m;
    size_t n=mxB->n;
    Simple_mat *mxA =new Simple_mat(m,n);
    double *mxA_ptr=mxA->ptr;
    double *mxB_ptr=mxB->ptr;
    size_t e_num = m*n;  
    for(size_t i(0);i<e_num;++i){
        mxA_ptr[i]=mxB_ptr[i];
    }
    return mxA;
}


inline void copy_mxArray_2_vector(const mxArray* mxA, vector<size_t> & vec){
    double *mxA_ptr=mxGetPr(mxA);
    size_t e_num = mxGetNumberOfElements(mxA);  
    vec.clear();
    for(size_t i(0);i<e_num;++i){
        vec.push_back((size_t)round(mxA_ptr[i]));
    }
}

inline mxArray * create_mxArray_from_vector(const vector<size_t> & vec){
    mxArray * mxC=mxCreateDoubleMatrix(vec.size(), 1, mxREAL);
    double *mxC_ptr=mxGetPr(mxC);
    size_t e_num = mxGetNumberOfElements(mxC);  
    for(size_t i(0);i<e_num;++i){
        mxC_ptr[i]=vec[i];
    }
    return mxC;
}


inline void get_mat_col_char(const char * mxA_ptr, const size_t &col_idx, char *mxC_ptr, const size_t &row_num){
    size_t st_idx=col_idx*row_num;
    for (size_t i(0); i<row_num; ++i){
        mxC_ptr[i]=mxA_ptr[st_idx+i];
    }
}


inline void get_mat_col(const mxArray * mxA, const size_t &col_idx, mxArray* mxC){
    size_t row_num=mxGetM(mxA);
    double *mxC_ptr=mxGetPr(mxC);
    double *mxA_ptr=mxGetPr(mxA);
    size_t st_idx=col_idx*row_num;
    for (size_t i(0); i<row_num; ++i){
        mxC_ptr[i]=mxA_ptr[st_idx+i];
    }
}

inline void get_mat_row(const mxArray * mxA, const size_t &row_idx, mxArray* mxC){
    size_t col_num=mxGetN(mxA);
    size_t row_num=mxGetM(mxA);
    double *mxC_ptr=mxGetPr(mxC);
    double *mxA_ptr=mxGetPr(mxA);
    size_t st_idx=row_idx;
    for (size_t i(0); i<col_num; ++i){
        mxC_ptr[i]=mxA_ptr[st_idx+i*row_num];
    }
}

inline Simple_mat * copy_mat_row(Simple_mat * mxA, const size_t &row_idx){
    size_t col_num=mxA->n;
    size_t row_num=mxA->m;
    Simple_mat *mxC=new Simple_mat(1, col_num);
    double *mxC_ptr=mxC->ptr;
    double *mxA_ptr=mxA->ptr;
    size_t st_idx=row_idx;
    for (size_t i(0); i<col_num; ++i){
        mxC_ptr[i]=mxA_ptr[st_idx+i*row_num];
    }
    return mxC;
}


inline void get_mat_row(const Simple_mat &mxA, const size_t &row_idx, Simple_mat &mxC){
    size_t col_num=mxA.n;
    size_t row_num=mxA.m;
    double *mxC_ptr=mxC.ptr;
    double *mxA_ptr=mxA.ptr;
    size_t st_idx=row_idx;
    for (size_t i(0); i<col_num; ++i){
        mxC_ptr[i]=mxA_ptr[st_idx+i*row_num];
    }
}

inline void get_mat_row(const mxArray * mxA, const size_t &row_idx, Simple_mat &mxC){
    size_t col_num=mxGetN(mxA);
    size_t row_num=mxGetM(mxA);
    double *mxC_ptr=mxC.ptr;
    double *mxA_ptr=mxGetPr(mxA);
    size_t st_idx=row_idx;
    for (size_t i(0); i<col_num; ++i){
        mxC_ptr[i]=mxA_ptr[st_idx+i*row_num];
    }
}

//TODO: need to verify
inline void get_mat_row_sparseA(const mxArray * mxA, const size_t &row_idx, Simple_mat &mxC){
            
    size_t col_num=mxGetN(mxA);
    double *mxC_ptr=mxC.ptr;
    double *mxA_ptr=mxGetPr(mxA);
    mwIndex *ir=mxGetIr(mxA);
    mwIndex *jc=mxGetJc(mxA);
    size_t st_idx=row_idx;
    for (size_t col_idx(0); col_idx<col_num; ++col_idx){
        int one_col_first_e_idx=jc[col_idx];
        int one_col_last_e_idx=jc[col_idx+1]-1;
        double one_e_v(0);
        if (one_col_first_e_idx<=one_col_last_e_idx){
            size_t one_col_first_e_row_idx=ir[one_col_first_e_idx];
            if (row_idx>=one_col_first_e_row_idx){
                size_t one_col_last_e_row_idx=ir[one_col_last_e_idx];
                if (row_idx<=one_col_last_e_row_idx){
                    for (int e_idx=one_col_first_e_idx; e_idx<=one_col_last_e_idx; ++e_idx){
                        size_t test_row_idx=ir[e_idx];
                        if (test_row_idx==row_idx){
                            one_e_v=mxA_ptr[e_idx];
                            break;
                        }
                    }
                }
            }
        }
        mxC_ptr[col_idx]=one_e_v;
    }
}

//TODO: need to verify
inline void set_mat_row(const mxArray * mxA, const size_t &row_idx, const mxArray* mxC){
    size_t col_num=mxGetN(mxA);
    size_t row_num=mxGetM(mxA);
    double *mxC_ptr=mxGetPr(mxC);
    double *mxA_ptr=mxGetPr(mxA);
    size_t st_idx=row_idx;
    for (size_t i(0); i<col_num; ++i){
        mxA_ptr[st_idx+i*row_num]=mxC_ptr[i];
    }
}

inline void set_mat_row(Simple_mat * mxA, const size_t &row_idx, Simple_mat *mxC){
    size_t col_num=mxA->n;
    size_t row_num=mxA->m;
    double *mxC_ptr=mxC->ptr;
    double *mxA_ptr=mxA->ptr;
    size_t st_idx=row_idx;
    for (size_t i(0); i<col_num; ++i){
        mxA_ptr[st_idx+i*row_num]=mxC_ptr[i];
    }
}

inline void set_mat_col(Simple_mat * mxA, const size_t &col_idx, Simple_mat *mxC){
    size_t col_num=mxA->n;
    size_t row_num=mxA->m;
    double *mxC_ptr=mxC->ptr;
    double *mxA_ptr=mxA->ptr;
    size_t st_idx=col_idx*row_num;
    for (size_t i(0); i<row_num; ++i){
        mxA_ptr[st_idx+i]=mxC_ptr[i];
    }
}


inline void print_mat_bool(const mxArray* mxA){
    
    if (!mxA){
        printf("\n------------------------ mat is not valid! \n");
        return;
    }
    
    size_t m = mxGetM(mxA);  
    size_t n = mxGetN(mxA);  
    
    printf("\n------------------------ \n");
    printf("\n m:%d, n:%d \n", m,n);
    
    if (mxIsSparse(mxA)){
        printf("\n------------------------ ERROR: cannot print sparse matrix \n");
        return;
    }
    
    char *mxA_ptr=(char*)mxGetData(mxA);
    for(size_t i(0);i<m;++i){
        for(size_t j(0);j<n;++j){
            if (mxA_ptr[j*m+i]){
                printf(" 1 ");  
            }else{
                printf(" 0 ");  
            }
        }
        printf("\n");
    }
    printf("------------------------ \n");
}



inline void print_mat(const mxArray* mxA){
    
    if (!mxA){
        printf("\n------------------------ mat is not valid! \n");
        return;
    }
    
    size_t m = mxGetM(mxA);  
    size_t n = mxGetN(mxA);  
    
    printf("\n------------------------ \n");
    printf("\n m:%d, n:%d \n", m,n);
    
    if (mxIsSparse(mxA)){
        printf("\n------------------------ ERROR: cannot print sparse matrix \n");
        return;
    }
    
    double *mxA_ptr=mxGetPr(mxA);
    for(size_t i(0);i<m;++i){
        for(size_t j(0);j<n;++j){
           printf(" %.4f ", mxA_ptr[j*m+i]);  
        }
        printf("\n");
    }
    printf("------------------------ \n");
}



inline void print_mat(Simple_mat * mxA){
    
    if (!mxA){
        printf("\n------------------------ mat is not valid! \n");
        return;
    }
    
    size_t m = mxA->m;  
    size_t n = mxA->n;  
    printf("\n------------------------ \n");
    printf("\n m:%d, n:%d \n", m,n);
    
    if (mxA->sparse){
        printf("\n------------------------ ERROR: cannot print sparse matrix \n");
        return;
    }
    
    double *mxA_ptr=mxA->ptr;
    for(size_t i(0);i<m;++i){
        for(size_t j(0);j<n;++j){
           printf(" %.4f ", mxA_ptr[j*m+i]);  
        }
        printf("\n");
    }
    printf("------------------------ \n");
}


inline mxArray * create_pos_sel_mat(const mxArray* mxA){
    
    double ZERO_EPS(1e-10);
    
    size_t m = mxGetM(mxA);  
    size_t n = mxGetN(mxA);  
    mxArray * mxC=mxCreateDoubleMatrix(m, n, mxREAL);
    double *mxA_ptr=mxGetPr(mxA);
    double *mxC_ptr=mxGetPr(mxC);
    size_t e_num=m*n;
    for(size_t i(0);i<e_num;++i){
        if(mxA_ptr[i]>ZERO_EPS){
            mxC_ptr[i]=1;
        }else{
            mxC_ptr[i]=0;
        }
    }
    return mxC;
}

inline mxArray * create_neg_sel_mat(const mxArray* mxA){
    
    double ZERO_EPS(1e-10);
    
    size_t m = mxGetM(mxA);  
    size_t n = mxGetN(mxA);  
    mxArray * mxC=mxCreateDoubleMatrix(m, n, mxREAL);
    double *mxA_ptr=mxGetPr(mxA);
    double *mxC_ptr=mxGetPr(mxC);
    size_t e_num=m*n;
    for(size_t i(0);i<e_num;++i){
        if(mxA_ptr[i]<-ZERO_EPS){
            mxC_ptr[i]=1;
        }else{
            mxC_ptr[i]=0;
        }
    }
    return mxC;
}


inline char * create_pos_sel_mat2(const mxArray* mxA){
    
    double ZERO_EPS(1e-10);
    
    size_t m = mxGetM(mxA);  
    size_t n = mxGetN(mxA);  
    
    double *mxA_ptr=mxGetPr(mxA);
    char *mxC_ptr=new char[m*n];
    size_t e_num=m*n;
    for(size_t i(0);i<e_num;++i){
        if(mxA_ptr[i]>ZERO_EPS){
            mxC_ptr[i]=1;
        }else{
            mxC_ptr[i]=0;
        }
    }
    return mxC_ptr;
}

inline char * create_neg_sel_mat2(const mxArray* mxA){
    
    double ZERO_EPS(1e-10);
    
    size_t m = mxGetM(mxA);  
    size_t n = mxGetN(mxA);  
    
    double *mxA_ptr=mxGetPr(mxA);
    char *mxC_ptr=new char[m*n];
    size_t e_num=m*n;
    for(size_t i(0);i<e_num;++i){
        if(mxA_ptr[i]<-ZERO_EPS){
            mxC_ptr[i]=1;
        }else{
            mxC_ptr[i]=0;
        }
    }
    return mxC_ptr;
}


#endif /*CP_MATLIB_H_*/

