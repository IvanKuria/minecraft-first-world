/* Minimal pthread shim for building the unmodified CPU reference (c5search.c) with MSVC. */
#pragma once
#include <windows.h>
#include <process.h>
typedef HANDLE pthread_t;
typedef CRITICAL_SECTION pthread_mutex_t;
static INIT_ONCE c5_once = INIT_ONCE_STATIC_INIT;
#define PTHREAD_MUTEX_INITIALIZER {0}
static BOOL CALLBACK c5_init_cs(PINIT_ONCE o, PVOID p, PVOID *c) { (void)o; (void)c; InitializeCriticalSection((CRITICAL_SECTION *)p); return TRUE; }
static int pthread_mutex_lock(pthread_mutex_t *m) { InitOnceExecuteOnce(&c5_once, c5_init_cs, m, NULL); EnterCriticalSection(m); return 0; }
static int pthread_mutex_unlock(pthread_mutex_t *m) { LeaveCriticalSection(m); return 0; }
typedef struct { void *(*fn)(void *); void *arg; } c5_start;
static unsigned __stdcall c5_tramp(void *p) { c5_start s = *(c5_start *)p; free(p); s.fn(s.arg); return 0; }
static int pthread_create(pthread_t *t, void *attr, void *(*fn)(void *), void *arg) {
    (void)attr; c5_start *s = (c5_start *)malloc(sizeof *s); s->fn = fn; s->arg = arg;
    *t = (HANDLE)_beginthreadex(NULL, 0, c5_tramp, s, 0, NULL); return *t ? 0 : 1; }
static int pthread_join(pthread_t t, void **r) { (void)r; WaitForSingleObject(t, INFINITE); CloseHandle(t); return 0; }
