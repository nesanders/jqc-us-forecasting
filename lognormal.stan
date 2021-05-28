data {
    int<lower=0> N; // Number of events
    vector[N] y; // Number of days since event
    int<lower=0> y_min; // minimum y-value
}
transformed data {
    int<lower=0> N_gte_10=0; // Number of events at y>=10
    for (n in 1:N) {
        if (y[n] >= 10) {N_gte_10 += 1;}
    }
}
parameters {
    real<lower=0> sigma;
    real<lower=0> mu;
}
model {
    for (i in 1:N) {
        y[i] ~ lognormal(mu, sigma) T[y_min, ];
    }
    mu ~ cauchy(0,1);
    sigma ~ cauchy(0,1);
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
            log_likelihood[n] = lognormal_lpdf(y[n] | mu, sigma) - lognormal_lccdf(y_min| mu, sigma);
        }
        if (y[n] >= 10) {
            log_likelihood_gte_10[n_gte_10] = log_likelihood[n];
            n_gte_10 += 1;
        }
    }
    }
}
