//! CPython's `random.Random`: the Mersenne Twister (MT19937) seeded from an
//! integer as CPython seeds it, with `random()`, `shuffle` and the integer
//! draws they rest on.
//!
//! The protocols this harness measures fix their sample sets and arrival
//! sequences by a seed: ShareGPT's, MJHQ's, Beans', UEval's and PIE-Bench's
//! rows are the first of a seeded shuffle, and a Poisson load's intervals are
//! seeded draws. Drawing them as CPython does keeps those protocols' inputs the
//! same under every implementation of the harness.

const N: usize = 624;
const M: usize = 397;

/// A generator in CPython's `random.Random` state.
pub struct Random {
    state: [u32; N],
    index: usize,
}

impl Random {
    /// `random.Random(seed)`: `init_by_array` of the seed's 32-bit words, least
    /// significant first (one zero word for a zero seed).
    pub fn new(seed: u64) -> Self {
        let mut key = vec![(seed & 0xffff_ffff) as u32];
        if seed >> 32 != 0 {
            key.push((seed >> 32) as u32);
        }
        let mut random = Self {
            state: [0; N],
            index: N,
        };
        random.init_by_array(&key);
        random
    }

    /// NumPy's legacy `np.random.seed(seed)` of a 32-bit seed, which arrival
    /// intervals are drawn from: `init_genrand`, without `init_by_array`.
    pub fn numpy(seed: u32) -> Self {
        let mut random = Self {
            state: [0; N],
            index: N,
        };
        random.init_genrand(seed);
        random
    }

    fn init_genrand(&mut self, seed: u32) {
        self.state[0] = seed;
        for index in 1..N {
            let previous = self.state[index - 1];
            self.state[index] = 1_812_433_253u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(index as u32);
        }
        self.index = N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19_650_218);
        let (mut i, mut j) = (1usize, 0usize);
        for _ in 0..N.max(key.len()) {
            let previous = self.state[i - 1];
            self.state[i] = (self.state[i] ^ (previous ^ (previous >> 30)).wrapping_mul(1_664_525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.state[0] = self.state[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..N - 1 {
            let previous = self.state[i - 1];
            self.state[i] = (self.state[i]
                ^ (previous ^ (previous >> 30)).wrapping_mul(1_566_083_941))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.state[0] = self.state[N - 1];
                i = 1;
            }
        }
        self.state[0] = 0x8000_0000;
    }

    /// One 32-bit draw: `genrand_uint32`.
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            for k in 0..N {
                let y = (self.state[k] & 0x8000_0000) | (self.state[(k + 1) % N] & 0x7fff_ffff);
                let mut next = self.state[(k + M) % N] ^ (y >> 1);
                if y & 1 != 0 {
                    next ^= 0x9908_b0df;
                }
                self.state[k] = next;
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `random()`: a float in `[0, 1)` from two draws, 53 bits of them.
    pub fn random(&mut self) -> f64 {
        let a = self.next_u32() >> 5;
        let b = self.next_u32() >> 6;
        (f64::from(a) * 67_108_864.0 + f64::from(b)) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// `getrandbits(k)` for `k` up to 64: whole words, least significant
    /// first, the last one's top bits.
    fn getrandbits(&mut self, bits: u32) -> u64 {
        if bits <= 32 {
            return u64::from(self.next_u32() >> (32 - bits));
        }
        let low = u64::from(self.next_u32());
        let high = u64::from(self.next_u32() >> (64 - bits));
        low | (high << 32)
    }

    /// `_randbelow(n)`: draws of `n`'s bit length until one is below `n`.
    pub fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        let bits = 64 - n.leading_zeros();
        loop {
            let draw = self.getrandbits(bits);
            if draw < n {
                return draw;
            }
        }
    }

    /// `shuffle(items)`: from the last position down to the second, each item
    /// swapped with one drawn from it and before it.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for index in (1..items.len()).rev() {
            let other = self.below(index as u64 + 1) as usize;
            items.swap(index, other);
        }
    }

    /// NumPy's legacy `exponential(scale)`: `scale * -log(1 - random())`.
    pub fn exponential(&mut self, scale: f64) -> f64 {
        scale * -(1.0 - self.random()).ln()
    }

    /// `expovariate(rate)`: `-log(1 - random()) / rate`.
    pub fn expovariate(&mut self, rate: f64) -> f64 {
        -(1.0 - self.random()).ln() / rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CPython 3.12: `random.Random(42)`'s first draws and its shuffle of ten.
    #[test]
    fn draws_as_cpython_draws() {
        let mut random = Random::new(42);
        assert_eq!(random.next_u32(), 2_746_317_213);
        let mut random = Random::new(42);
        assert!((random.random() - 0.639_426_798_457_883_8).abs() < 1e-15);
        let mut random = Random::new(42);
        let mut items: Vec<u32> = (0..10).collect();
        random.shuffle(&mut items);
        assert_eq!(items, [7, 3, 2, 8, 5, 6, 9, 4, 0, 1]);
        let mut random = Random::new(7);
        assert!((random.expovariate(16.0) - 0.024_457_177_764_675_268).abs() < 1e-15);
    }

    /// NumPy's legacy generator: `np.random.seed(42)`, then two
    /// `exponential(1 / 16)` draws.
    #[test]
    fn draws_as_numpy_draws() {
        let mut random = Random::numpy(42);
        assert!((random.exponential(1.0 / 16.0) - 0.029_329_255_623_553_693).abs() < 1e-15);
        assert!((random.exponential(1.0 / 16.0) - 0.188_132_589_432_345_07).abs() < 1e-15);
    }
}
