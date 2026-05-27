#ifndef FFT_H_
#define FFT_H_

#include "data.h"
#include "magia_tile_utils.h"
#include "magia_utils.h"
#include "fft_fs_params.h"

#define ALIGNMENT       (4)

/* Aligns the given address to 4-byte  */
#define ALIGN_4B(addr)  (((addr) + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1))

/* Division with round upwards */
#define DIV_UP(a, b)    (((a) + (b) - 1) / (b))

#define L1_BASE_TILE    (L1_BASE + (get_hartid() * L1_TILE_OFFSET))

#define FFT_PARAMS_BASE (L1_BASE_TILE)
#define FFT_PARAMS_SIZE ALIGN_4B(sizeof(fft_fs_params_t))

// DOG THE VEC LEN IS ACTUALLY DIVIDIE BY 2 AND THEN MULTIPLIED BY TWO, ITS A BYTE SIZE YO
#define CHUNK_AR_BASE   ALIGN_4B(FFT_PARAMS_BASE + FFT_PARAMS_SIZE)
#define CHUNK_AR_SIZE   ALIGN_4B(VEC_LEN)

#define CHUNK_AI_BASE   ALIGN_4B(CHUNK_AR_BASE + CHUNK_AR_SIZE)
#define CHUNK_AI_SIZE   ALIGN_4B(VEC_LEN)

#define CHUNK_BR_BASE   ALIGN_4B(CHUNK_AI_BASE + CHUNK_AI_SIZE)
#define CHUNK_BR_SIZE   ALIGN_4B(VEC_LEN)

#define CHUNK_BI_BASE   ALIGN_4B(CHUNK_BR_BASE + CHUNK_BR_SIZE)
#define CHUNK_BI_SIZE   ALIGN_4B(VEC_LEN)

#define CHUNK_WR_BASE   ALIGN_4B(CHUNK_BI_BASE + CHUNK_BI_SIZE)
#define CHUNK_WR_SIZE   ALIGN_4B(TW_LEN * 2)

#define CHUNK_WI_BASE   ALIGN_4B(CHUNK_WR_BASE + CHUNK_WR_SIZE)
#define CHUNK_WI_SIZE   ALIGN_4B(TW_LEN * 2)

#define CHUNK_CR_BASE   ALIGN_4B(CHUNK_WI_BASE + CHUNK_WI_SIZE)
#define CHUNK_CR_SIZE   ALIGN_4B(VEC_LEN)

#define CHUNK_DR_BASE   ALIGN_4B(CHUNK_CR_BASE + CHUNK_CR_SIZE)
#define CHUNK_DR_SIZE   ALIGN_4B(VEC_LEN)

#define CHUNK_CI_BASE   ALIGN_4B(CHUNK_DR_BASE + CHUNK_DR_SIZE)
#define CHUNK_CI_SIZE   ALIGN_4B(VEC_LEN)

#define CHUNK_DI_BASE   ALIGN_4B(CHUNK_CI_BASE + CHUNK_CI_SIZE)
#define CHUNK_DI_SIZE   ALIGN_4B(VEC_LEN)

#endif /* FFT_FS_H_ */
