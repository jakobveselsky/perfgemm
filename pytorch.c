template <typename scalar_t, typename opmath_t>
void gemm_notrans_(
	int64_t m, int64_t n, int64_t k,
	opmath_t alpha,
	const scalar_t *a, int64_t lda,
	const scalar_t *b, int64_t ldb,
	opmath_t beta,
	scalar_t *c, int64_t ldc) {
	// c *= beta
	scale_(m, n, beta, c, ldc);

	// c += alpha * (a @ b)
	for (const auto l : c10::irange(k)) {
		for (const auto j : c10::irange(n)) {
			opmath_t val = b[l + j * ldb] * alpha;
	      		int64_t i_m = m / 4;
			for (const auto i_i : c10::irange(i_m)) {
	        		c[j * ldc + i_i * 4 + 0] += a[i_i * 4 + 0 + l * lda] * val;
	        		c[j * ldc + i_i * 4 + 1] += a[i_i * 4 + 1 + l * lda] * val;
	        		c[j * ldc + i_i * 4 + 2] += a[i_i * 4 + 2 + l * lda] * val;
	        		c[j * ldc + i_i * 4 + 3] += a[i_i * 4 + 3 + l * lda] * val;
	      		}
	      		int64_t i = i_m * 4;
	      	for (; i < m; i++)
	        	c[j * ldc + i] += a[i + l * lda] * val;
	    	}
	}
}
