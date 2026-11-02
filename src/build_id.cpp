// Build identity of this library: the git hash the build compiled (+ "-dirty"), or the -DQE_GIT_HASH override;
// generated at build time into <build>/gen/qe_build_id.h (CMakeLists.txt, target qe_build_id), so only this file
// recompiles when it changes. Used by the Flash-Next [box] start-up line; the server reads the same value from
// build/qe_build.txt. Hidden: the libraries export only their C APIs.
#include "qe_build_id.h"

extern "C" __attribute__((visibility("hidden"))) const char* qe_internal_build_id() { return QE_BUILD_ID; }
