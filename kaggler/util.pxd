cdef inline double fmax(double a, double b): return a if a >= b else b
cdef inline double fmin(double a, double b): return a if a <= b else b

cdef double sigm(double x)
