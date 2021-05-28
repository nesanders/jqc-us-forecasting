data {
    int<lower=0> N; // Number of values
    vector[N] y; // values to evaluate
    int<lower=0> y_min; // minimum y-value
    real<lower=0> alpha;
    real<lower=0> sigma;
}
generated quantities {
    vector[N] p;
    
    for (n in 1:N) {
        p[n] = weibull_lpdf(y[n] | alpha, sigma) T[y_min, ];
    }
}
