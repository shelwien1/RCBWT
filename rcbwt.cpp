// rcbwt: BWT prototype (see RC_BWT.md).
// Stage now: only the "dumb BWT" reference via std::sort on cyclic rotations.
// argv[1] = input, argv[2] = streaming-BWT output (not yet implemented, left empty),
// argv[3] = reference BWT output (this file).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s input out bwt\n", argv[0]);
        return 1;
    }

    FILE* fin = fopen(argv[1], "rb");
    if (!fin) { perror(argv[1]); return 1; }
    fseek(fin, 0, SEEK_END);
    long sz = ftell(fin);
    fseek(fin, 0, SEEK_SET);
    if (sz <= 0) { fclose(fin); fprintf(stderr, "empty input\n"); return 1; }
    size_t n = (size_t)sz;

    std::vector<uint8_t> inp(n);
    if (fread(inp.data(), 1, n, fin) != n) { perror("fread"); fclose(fin); return 1; }
    fclose(fin);

    // Placeholder: streaming BWT not implemented yet -> create empty file.
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror(argv[2]); return 1; }
    fclose(fout);

    // Dumb BWT: std::sort the rotation indices using memcmp on a doubled buffer.
    std::vector<uint8_t> data2(2 * n);
    memcpy(data2.data(),     inp.data(), n);
    memcpy(data2.data() + n, inp.data(), n);

    std::vector<uint32_t> sa(n);
    for (size_t i = 0; i < n; i++) sa[i] = (uint32_t)i;

    const uint8_t* d = data2.data();
    std::sort(sa.begin(), sa.end(), [d, n](uint32_t a, uint32_t b) {
        return memcmp(d + a, d + b, n) < 0;
    });

    std::vector<uint8_t> bwt(n);
    for (size_t i = 0; i < n; i++) {
        bwt[i] = inp[(sa[i] + n - 1) % n];
    }

    FILE* fbwt = fopen(argv[3], "wb");
    if (!fbwt) { perror(argv[3]); return 1; }
    if (fwrite(bwt.data(), 1, n, fbwt) != n) { perror("fwrite"); fclose(fbwt); return 1; }
    fclose(fbwt);

    return 0;
}
