data {
    int<lower=0> N; // Number of events
    vector[N] y; // Number of fatalities per event
    int<lower=0> y_min; // minimum y-value
}
transformed data {
    int<lower=0> N_gte_10=0; // Number of events at y>=10
    for (n in 1:N) {
        if (y[n] >= 10) {N_gte_10 += 1;}
    }
}
parameters {
    real<lower=0> alpha;
    real<lower=0> sigma;
}
model {
    for (n in 1:N) {
        y[n] ~ weibull(alpha, sigma) T[y_min, ];
    }
    alpha ~ normal(0, 1);
    sigma ~ normal(0, 1);
}
generated quantities {
    vector[N] log_likelihood;
    vector[N_gte_10] log_likelihood_gte_10;
    
    {
    int n_gte_10 = 1;
    for (n in 1:N) {
        if (y[n] < y_min) {
            log_likelihood[n] = negative_infinity();
        } else {
            log_likelihood[n] = weibull_lpdf(y[n] | alpha, sigma) - weibull_lccdf(y_min| alpha, sigma);
        }
        if (y[n] >= 10) {
            log_likelihood_gte_10[n_gte_10] = log_likelihood[n];
            n_gte_10 += 1;
        }
    }
    }
}
