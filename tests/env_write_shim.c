// Logs every environment write of a process with the number of threads alive at that moment (issue #11: a write with
// more than one thread can crash a concurrent getenv). No GPU needed for the shim itself.
//   gcc -shared -fPIC -O1 tests/env_write_shim.c -o /tmp/env_write_shim.so -ldl
//   LD_PRELOAD=/tmp/env_write_shim.so flash_run ... 2>&1 | grep '^\[envw\]'
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

static int threads(void) {  // (raw reads: no stdio, no getenv inside a write)
    char b[4096];
    const int fd = (int)syscall(SYS_open, "/proc/self/status", 0);
    if (fd < 0) return -1;
    const long n = read(fd, b, sizeof b - 1);
    close(fd);
    if (n <= 0) return -1;
    b[n] = 0;
    const char* p = strstr(b, "Threads:");
    if (!p) return -1;
    int t = -1;
    sscanf(p + 8, "%d", &t);
    return t;
}
static void note(const char* fn, const char* what) {
    dprintf(2, "[envw] %s %s threads=%d tid=%ld pid=%d\n", fn, what ? what : "", threads(), (long)syscall(SYS_gettid), getpid());
}

int setenv(const char* n, const char* v, int o) {
    static int (*f)(const char*, const char*, int);
    if (!f) f = (int (*)(const char*, const char*, int))dlsym(RTLD_NEXT, "setenv");
    note("setenv", n);
    return f(n, v, o);
}
int unsetenv(const char* n) {
    static int (*f)(const char*);
    if (!f) f = (int (*)(const char*))dlsym(RTLD_NEXT, "unsetenv");
    note("unsetenv", n);
    return f(n);
}
int putenv(char* s) {
    static int (*f)(char*);
    if (!f) f = (int (*)(char*))dlsym(RTLD_NEXT, "putenv");
    note("putenv", s);
    return f(s);
}
int clearenv(void) {
    static int (*f)(void);
    if (!f) f = (int (*)(void))dlsym(RTLD_NEXT, "clearenv");
    note("clearenv", "");
    return f();
}
