int main() {
	/* gemm_notrans_(int m, int n, int k,
        float alpha,
        const float *a, int lda,
        const float *b, int ldb,
        float beta,
        float *c, int ldc) */
	return 0;
}

void scale_(int m, int n, float alpha, float *a, int lda) {
  if (alpha == 1.0) {
    return;  // identity
  }
  if (alpha == 0.0) {
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < m; i++) {
        a[j * lda + i] = 0.0;
      }
    }
    return;
  }
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      a[j * lda + i] *= alpha;
    }
  }
}

void gemm_notrans_(
	int m, int n, int k,
	float alpha,
	const float *a, int lda,
	const float *b, int ldb,
	float beta,
	float *c, int ldc) {
	// c *= beta
	scale_(m, n, beta, c, ldc);

	// c += alpha * (a @ b)
	for (int l; l < k; l++) {
		for (int j; j < n; j++) {
			float val = b[l + j * ldb] * alpha;
	      		int i_m = m / 4;
			for (int i_i = 0; i_i < i_m; i_i++) {
	        		c[j * ldc + i_i * 4 + 0] += a[i_i * 4 + 0 + l * lda] * val;
	        		c[j * ldc + i_i * 4 + 1] += a[i_i * 4 + 1 + l * lda] * val;
	        		c[j * ldc + i_i * 4 + 2] += a[i_i * 4 + 2 + l * lda] * val;
	        		c[j * ldc + i_i * 4 + 3] += a[i_i * 4 + 3 + l * lda] * val;
	      		}
	      		int i = i_m * 4;
	      	for (; i < m; i++)
	        	c[j * ldc + i] += a[i + l * lda] * val;
	    	}
	}
}
