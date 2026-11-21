#include <stddef.h>
#include <stdint.h>

static void pp_mmio_barrier(void)
{
#if defined(__arm__) || defined(__aarch64__)
    __asm__ __volatile__("dmb sy" ::: "memory");
#endif
}

/* AXI-Lite registers require individual aligned 32-bit bus accesses. */
void pp_mmio_write32(volatile uint32_t *mmio, const uint32_t *source, size_t count)
{
    for (size_t word = 0; word < count; ++word) {
        mmio[word] = source[word];
    }
    pp_mmio_barrier();
}

void pp_mmio_read32(uint32_t *destination, const volatile uint32_t *mmio, size_t count)
{
    pp_mmio_barrier();
    for (size_t word = 0; word < count; ++word) {
        destination[word] = mmio[word];
    }
}
