# BGESS

AVX2 microkernels for binary inner products and matrix multiplications.

Example usage on OS X

```
mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=RELEASE -mavx2 .. && make -j8 && VECLIB_MAXIMUM_THREADS=1 ./bgess --benchmark_min_time=0.1)
```
