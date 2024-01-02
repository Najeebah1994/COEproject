/* vec.c
 *
 * Author:
 * Date  :
 *
 *  Description
 */

/* Standard C includes */
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <stdio.h>

/*  -> SIMD header file  */
#include <immintrin.h>

/* Include common headers */
#include "common/macros.h"
#include "common/types.h"
#include "common/vmath.h"

/* Include application-specific headers */
#include "include/types.h"

/* Alternative Implementation */
#define inv_sqrt_2xPI 0.39894228040143270286


__m256 CNDF_v (__m256 InputX) 
{
 
 // Check for negative value of InputX
    __m256 negInputX = _mm256_mul_ps(InputX, _mm256_set1_ps(-1.0f));
    // Compare vectors using less-than operation
    __m256 inputmask = _mm256_cmp_ps(InputX, _mm256_set1_ps(0.0f), _CMP_LT_OS);
    __m256 xInput = _mm256_blendv_ps(InputX,negInputX, inputmask);
 
    // Compute NPrimeX term common to both four & six decimal accuracy calcs
    __m256 expValues= _mm256_mul_ps(xInput,xInput);
    expValues= _mm256_mul_ps(expValues,_mm256_set1_ps(-.5f));
    __m256 xNPrimeofX = _mm256_exp_ps(expValues);
    xNPrimeofX = _mm256_mul_ps(xNPrimeofX , _mm256_set1_ps(inv_sqrt_2xPI));

    __m256 xK2 = _mm256_mul_ps(_mm256_set1_ps(0.2316419f) , xInput);
    xK2 = _mm256_add_ps(_mm256_set1_ps(1.0f) ,xK2);
    xK2 = _mm256_div_ps(_mm256_set1_ps(1.0f) , xK2);
    __m256 xK2_2 = _mm256_mul_ps(xK2 , xK2);
    __m256 xK2_3 = _mm256_mul_ps(xK2_2 , xK2);
    __m256 xK2_4 = _mm256_mul_ps(xK2_3 , xK2);
    __m256 xK2_5 = _mm256_mul_ps(xK2_4 , xK2);
    
    __m256 xLocal_1 = _mm256_mul_ps(xK2 , _mm256_set1_ps(0.319381530f));
    __m256 xLocal_2 = _mm256_mul_ps(xK2_2 , _mm256_set1_ps(-0.356563782f));
    __m256 xLocal_3 = _mm256_mul_ps(xK2_3 , _mm256_set1_ps(1.781477937f));
    xLocal_2 = _mm256_add_ps(xLocal_2 , xLocal_3);
    xLocal_3 = _mm256_mul_ps(xK2_4 ,_mm256_set1_ps(-1.821255978f));
    xLocal_2 = _mm256_add_ps(xLocal_2 , xLocal_3);
    xLocal_3 = _mm256_mul_ps(xK2_5 , _mm256_set1_ps(1.330274429f));
    xLocal_2 = _mm256_add_ps(xLocal_2 , xLocal_3);

    xLocal_1 = _mm256_add_ps(xLocal_2 , xLocal_1);
    __m256 xLocal   = _mm256_mul_ps(xLocal_1 , xNPrimeofX);
    xLocal   = _mm256_sub_ps(_mm256_set1_ps(1.0f) , xLocal);

    __m256 NegOutputX  = _mm256_sub_ps(_mm256_set1_ps(1.0f),xLocal);
    __m256 OutputX= _mm256_blendv_ps(xLocal,NegOutputX,inputmask);
    
    return OutputX;
} 


__m256 blackScholes_v(__m256 sptprice, __m256 strike, __m256 rate, __m256 volatility,
                   __m256 otime, __m256 otype)
{
    __m256 xSqrtTime = _mm256_sqrt_ps(otime);
    __m256 logValues = _mm256_div_ps(sptprice , strike);
    __m256 xLogTerm = _mm256_log_ps(logValues); 
    __m256 xPowerTerm = _mm256_mul_ps(volatility , volatility);
    xPowerTerm = _mm256_mul_ps(xPowerTerm ,_mm256_set1_ps(.5f));
        
    __m256 xD1 = _mm256_add_ps(rate,xPowerTerm);
    xD1 = _mm256_mul_ps(xD1 , otime);
    xD1 = _mm256_add_ps(xD1 , xLogTerm);

    __m256 xDen = _mm256_mul_ps(volatility , xSqrtTime);
    xD1 = _mm256_div_ps(xD1 ,xDen);
    __m256 xD2 = _mm256_sub_ps(xD1, xDen);

    __m256 d1 = _mm256_mul_ps(xD1 ,_mm256_set1_ps(1.0f)); 
    __m256 d2 = _mm256_mul_ps(xD2 ,_mm256_set1_ps(1.0f));
    __m256 NofXd1 = CNDF_v(d1);
    __m256 NofXd2 = CNDF_v(d2);

    __m256 FutureValueX = _mm256_mul_ps(rate,otime);
    FutureValueX= _mm256_mul_ps(FutureValueX, _mm256_set1_ps(-1.0f));
    FutureValueX= _mm256_exp_ps(FutureValueX);
    FutureValueX= _mm256_mul_ps(FutureValueX, strike);

    __m256 OptionPrice01= _mm256_mul_ps(FutureValueX,NofXd2);
    __m256 OptionPrice02= _mm256_mul_ps(sptprice,NofXd1);
    __m256 OptionPrice0= _mm256_sub_ps(OptionPrice02,OptionPrice01);
 
    __m256 NegNofXd1 = _mm256_sub_ps(_mm256_set1_ps(1.0f) , NofXd1);
    __m256 NegNofXd2 = _mm256_sub_ps(_mm256_set1_ps(1.0f) , NofXd2); 
    __m256 OptionPrice11 = _mm256_mul_ps(sptprice,NegNofXd1);
    __m256 OptionPrice12 = _mm256_mul_ps(FutureValueX,NegNofXd2);
    __m256 OptionPrice1 = _mm256_sub_ps(OptionPrice12,OptionPrice11);

    //__m256 otype1_vm= _mm256_castsi256_ps(otype);
    __m256 otype_vm = _mm256_cmp_ps(otype, _mm256_setzero_ps(), _CMP_EQ_OQ);
    __m256 mask = _mm256_blendv_ps(_mm256_setzero_ps(), otype_vm,otype_vm);
    __m256 OptionPrice_f = _mm256_blendv_ps(OptionPrice1,OptionPrice0,mask);
    return OptionPrice_f;
}


/* Alternative Implementation */
void* impl_vector(void* args)
{
    args_t* a = (args_t*) args;

    size_t i;
    size_t num_stocks = a->num_stocks;

    float* sptprice   = a->sptPrice  ;
    float* strike     = a->strike    ;
    float* rate       = a->rate      ;
    float* volatility = a->volatility;
    float* otime      = a->otime     ;
    char * otype      = a->otype     ;
    float* output     = a->output    ;

    __m256i vm = _mm256_set1_epi32(0x80000000);
    const int max_vlen = 32 / sizeof(float);

    for (register size_t hw_vlen, i = 0; i < num_stocks; i += hw_vlen) {
      register int rem = num_stocks - i;
      hw_vlen = rem < max_vlen ? rem : max_vlen;        /* num of elems      */
      if (hw_vlen < max_vlen)
      {
      unsigned int m[max_vlen];
      for (size_t j = 0; j < max_vlen; j++)
        m[j] = (j < hw_vlen) ? 0x80000000 : 0x00000000;
        vm = _mm256_setr_epi32(m[0], m[1], m[2], m[3],
                             m[4], m[5], m[6], m[7]);
      }

        __m256 sptprice_vec = _mm256_maskload_ps(sptprice,vm);
        __m256 strike_vec = _mm256_maskload_ps(strike,vm);
        __m256 rate_vec = _mm256_maskload_ps(rate,vm);
        __m256 volatility_vec = _mm256_maskload_ps(volatility,vm);
        __m256 otime_vec = _mm256_maskload_ps(otime,vm);
        __m256i otype_vec= _mm256_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (otype)));
        otype_vec= _mm256_cmpeq_epi32(otype_vec, _mm256_set1_epi32('P'));
        __m256 otype_vec1= _mm256_castsi256_ps(otype_vec);
        __m256 result = blackScholes_v(sptprice_vec, strike_vec, rate_vec, volatility_vec, otime_vec, otype_vec1);

        _mm256_maskstore_ps(output,vm,result);

        sptprice +=hw_vlen;
        strike +=hw_vlen;
        rate +=hw_vlen; 
        volatility +=hw_vlen;
        otime +=hw_vlen;
        otype +=hw_vlen;
        output +=hw_vlen;    
}
}