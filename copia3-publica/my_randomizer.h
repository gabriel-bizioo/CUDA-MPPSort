// Fully <random>-compatible, host+device safe, tiny and blazing fast
struct Xoroshiro128Plus {
    using result_type = uint64_t;                  // ← required

    uint64_t s[2];

    static constexpr result_type min() { return 0; }          // ← required
    static constexpr result_type max() { return UINT64_MAX; } // ← required

    Xoroshiro128Plus(uint64_t seed = 5489u) { this->seed(seed); }

    void seed(uint64_t seedval) {
        uint64_t z = (seedval += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        s[0] = z ^ (z >> 31);

        z = (seedval += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        s[1] = z ^ (z >> 31);
    }

    uint64_t operator()() {
        uint64_t x = s[0];
        uint64_t y = s[1];
        uint64_t result = x + y;
        y ^= x;
        s[0] = rotl(x, 24) ^ y ^ (y << 16);  // a, b
        s[1] = rotl(y, 37);                    // c
        return result;
    }

private:
    static inline constexpr uint64_t rotl(uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }
};
