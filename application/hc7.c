/* Hidden-cost model, one realisation, for Figure 7 (sport application).
   Same step order and rules as the Python model in core/ and simulation/:
   origination, propagation, recovery (rt = 1), failure realisation with the current protection level,
   capacity update, imitation of S0 then S1 with the Fermi rule on c_m (synchronous copy from a snapshot,
   role model drawn from the whole population), exploration of S0 then S1.
   Additions: regime flag (0 conditional: only failed nodes transmit; 1 unrestricted: every node transmits)
   and per-agent stationary means.
   Build: gcc -O3 -march=native -shared -fPIC -o libhc7.so hc7.c -lm                                   */
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static inline uint64_t rotl(const uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
typedef struct { uint64_t s[4]; int has_g; double g; } rng_t;
static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static void rng_seed(rng_t *r, uint64_t seed) {
    uint64_t x = seed;
    for (int i = 0; i < 4; i++) r->s[i] = splitmix64(&x);
    r->has_g = 0; r->g = 0.0;
}
static inline uint64_t rng_next(rng_t *r) {
    uint64_t *s = r->s;
    const uint64_t result = rotl(s[0] + s[3], 23) + s[0];
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
    return result;
}
static inline double rng_u(rng_t *r) { return (rng_next(r) >> 11) * 0x1.0p-53; }
static inline int rng_int(rng_t *r, int n) { return (int)(rng_u(r) * n); }
static inline double rng_normal(rng_t *r) {
    if (r->has_g) { r->has_g = 0; return r->g; }
    double u1 = rng_u(r), u2 = rng_u(r);
    while (u1 <= 1e-300) u1 = rng_u(r);
    double rad = sqrt(-2.0 * log(u1)), th = 6.283185307179586 * u2;
    r->g = rad * sin(th); r->has_g = 1;
    return rad * cos(th);
}

/* out[0..5]: stationary means of c_m, f, f_p, f_p*c (cost), SD of f_p across agents, p_p
   agent_out (may be NULL): 4 x n per-agent stationary means of f_p, failure, c_m, p_p          */
int run_realisation(int n, const int *src, const int *dst, int n_edges_dir,
                    const double *cent, double pe, double pr, long T,
                    double pn, double pl, double pmax, double cp, double cin, double fm,
                    double memory, double s, double sigma, uint64_t seed, int regime,
                    double *out, double *agent_out)
{
    rng_t R; rng_seed(&R, seed);
    double *Capital = malloc(n * sizeof(double)), *Cm = malloc(n * sizeof(double));
    double *S0 = calloc(n, sizeof(double)), *S1 = calloc(n, sizeof(double));
    double *fp = malloc(n * sizeof(double)), *pp = malloc(n * sizeof(double));
    double *tmp = malloc(n * sizeof(double));
    double *afp = calloc(n, sizeof(double)), *af = calloc(n, sizeof(double));
    double *acm = calloc(n, sizeof(double)), *app = calloc(n, sizeof(double));
    unsigned char *fail = calloc(n, 1), *pot = malloc(n);
    int *im = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) { Capital[i] = cin; Cm[i] = memory; }
    long T_half = T / 2;
    double acc[6] = {0, 0, 0, 0, 0, 0};
    long nacc = 0;

    for (long t = 0; t < T; t++) {
        for (int i = 0; i < n; i++) pot[i] = (rng_u(&R) < pn);                 /* 1 origination */
        for (int e = 0; e < n_edges_dir; e++)                                    /* 2 propagation */
            if ((regime == 1 || fail[src[e]]) && rng_u(&R) <= pl) pot[dst[e]] = 1;
        memset(fail, 0, n);                                                      /* 3 recovery, rt = 1 */
        double sfp = 0, sfp2 = 0, spp = 0;
        for (int i = 0; i < n; i++) {                                            /* 4 realisation */
            double v = S0[i] + S1[i] * cent[i];
            if (v < 0) v = 0; if (v > 1.0 - fm) v = 1.0 - fm;
            fp[i] = v;
            double fc = v * Capital[i]; if (fc < 1e-12) fc = 1e-12;
            pp[i] = pmax / (1.0 + cp / fc);
            spp += pp[i]; sfp += v; sfp2 += v * v;
            if (pot[i] && rng_u(&R) > pp[i]) { fail[i] = 1; Capital[i] = 0.0; }
        }
        double scm = 0, sf = 0, scost = 0;
        for (int i = 0; i < n; i++) {                                            /* 5 capacity update */
            scost += fp[i] * Capital[i];
            Capital[i] = cin + (1.0 - fm - fp[i]) * Capital[i];
            Cm[i] = memory * Capital[i] + (1.0 - memory) * Cm[i];
            scm += Cm[i]; sf += fail[i];
        }
        if (t >= T_half) {
            double mfp = sfp / n, var = sfp2 / n - mfp * mfp; if (var < 0) var = 0;
            acc[0] += scm / n; acc[1] += sf / n; acc[2] += mfp; acc[3] += scost / n;
            acc[4] += sqrt(var); acc[5] += spp / n; nacc++;
            if (agent_out) for (int i = 0; i < n; i++) { afp[i] += fp[i]; af[i] += fail[i]; acm[i] += Cm[i]; app[i] += pp[i]; }
        }
        double *strats[2] = {S0, S1};                                            /* 6, 7 imitation */
        for (int k = 0; k < 2; k++) {
            double *st = strats[k]; int nim = 0;
            for (int i = 0; i < n; i++) if (rng_u(&R) <= pr) im[nim++] = i;
            if (!nim) continue;
            memcpy(tmp, st, n * sizeof(double));
            for (int j = 0; j < nim; j++) {
                int i = im[j], rr = rng_int(&R, n);
                while (rr == i) rr = rng_int(&R, n);
                double x = -s * (Cm[rr] - Cm[i]); if (x > 500) x = 500; if (x < -500) x = -500;
                double pi = 1.0 / (1.0 + exp(x));
                if (rng_u(&R) <= pi) st[i] = tmp[rr];
            }
        }
        for (int i = 0; i < n; i++) if (rng_u(&R) <= pe) S0[i] += sigma * rng_normal(&R);   /* 8, 9 exploration */
        for (int i = 0; i < n; i++) if (rng_u(&R) <= pe) S1[i] += sigma * rng_normal(&R);
    }
    for (int k = 0; k < 6; k++) out[k] = acc[k] / nacc;
    if (agent_out) for (int i = 0; i < n; i++) {
        agent_out[i] = afp[i] / nacc; agent_out[n + i] = af[i] / nacc;
        agent_out[2 * n + i] = acm[i] / nacc; agent_out[3 * n + i] = app[i] / nacc;
    }
    free(Capital); free(Cm); free(S0); free(S1); free(fp); free(pp); free(tmp); free(fail); free(pot); free(im);
    free(afp); free(af); free(acm); free(app);
    return 0;
}
