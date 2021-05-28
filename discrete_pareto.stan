// See https://github.com/nesanders/stan-discrete-pareto

functions{
    real hurwitz_zeta(real s, real a);
}
data{
  int<lower=0> N; // number of datapoints
  int<lower=0> y_min; // minimum y-value
  int<lower=y_min> y[N]; // Datapoints
}
transformed data{
  int<lower=0> K = max(y) - min(y) + 1; // number of unique values
  int values[K]; // y-values
  int<lower=0> frequencies[K]; // number of counts at each y-value
  real real_ymin; // needed for C+ Hurwitz zeta
  int<lower=0> N_gte_10=0; // Number of events at y>=10
  
  for (k in 1:K) {
      values[k] = min(y) + k - 1;
      frequencies[k] = 0;
  }
  for (n in 1:N) {
      int k;
      k = y[n] - min(y) + 1;
      frequencies[k] += 1;
  }
  real_ymin = round(y_min);
  
  for (n in 1:N) {
      if (y[n] >= 10) {N_gte_10 += 1;}
  }
}
parameters{
  real <lower=1> alpha;
}
model{
  real constant = log(hurwitz_zeta(alpha, real_ymin));
  for (k in 1:K) {
    target += frequencies[k] * (-alpha * log(values[k]) - constant);
  }
  alpha ~ normal(2, 2);
} 
/*TODO: implement rng for prior and posterior predictive checks*/
generated quantities {
    vector[N] log_likelihood;
    vector[N_gte_10] log_likelihood_gte_10;
    
    {
    int n_gte_10 = 1;
    for (n in 1:N) {
        log_likelihood[n] = -alpha*log(y[n]) - log(hurwitz_zeta(alpha, real_ymin));
        if (y[n] >= 10) {
            log_likelihood_gte_10[n_gte_10] = log_likelihood[n];
            n_gte_10 += 1;
        }
    }
    }
}
