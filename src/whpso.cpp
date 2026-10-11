// WILD HEARTS shader pre-compiler (proxy dxgi.dll)
//
// Loaded by the game in place of the system dxgi.dll. It forwards every DXGI
// call to the real System32 dxgi.dll and hooks the ID3D12Device methods that
// build pipelines (CreateRootSignature, CreateGraphicsPipelineState,
// CreateComputePipelineState, CreatePipelineState). The hooks sit in the D3D12
// runtime, not the GPU driver, so they work the same on NVIDIA, AMD and Intel.
// Inside the game process:
//   * every pipeline the game builds is recorded to WH_SHADERS/cache
//   * at launch, every recorded pipeline is rebuilt exactly on background
//     threads, so the driver's shader cache holds them before the game asks
//   * then every shader the game ships is built too, using the root signature
//     and render state of pipelines the game itself built (precompile.inc).
//     Progress is kept per GPU and driver, so later launches only do new work.
//   * until that is done, a progress line is drawn in the top-left corner of
//     the game's frames (overlay.inc)
// Everything lives in a WH_SHADERS folder next to dxgi.dll:
//   WH_SHADERS/WH_PSO.ini   settings
//   WH_SHADERS/seed         pipelines shipped with the DLL (rebuilt like recorded ones)
//   WH_SHADERS/cache        recorded pipelines and pre-compile progress
//   WH_SHADERS/logs         the log, when logging is on
// If the game folder can't be written to, cache and logs go to
// %LOCALAPPDATA%\WH_SHADERS\<id of the game folder> instead.
// Delete dxgi.dll from the game folder to undo.
#define WIN32_LEAN_AND_MEAN
#define WIDL_EXPLICIT_AGGREGATE_RETURNS   // correct ABI for D3D12 methods that return structs
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <stdint.h>
#include <stdio.h>
#include <io.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <atomic>
#include <memory>
#include <algorithm>

#define WH_PSO_VERSION "1.12.0"   // bump for every release; the log restarts when the DLL changes

// ------------------------------------------------------------------ forwarding
#define DXGI_EXPORTS(X) \
    X(ApplyCompatResolutionQuirking) X(CompatString) X(CompatValue) \
    X(DXGID3D10CreateDevice) X(DXGID3D10CreateLayeredDevice) X(DXGID3D10GetLayeredDeviceSize) \
    X(DXGID3D10RegisterLayers) X(DXGIDeclareAdapterRemovalSupport) X(DXGIDisableVBlankVirtualization) \
    X(DXGIDumpJournal) X(DXGIGetDebugInterface1) X(DXGIReportAdapterConfiguration) \
    X(PIXBeginCapture) X(PIXEndCapture) X(PIXGetCaptureState) X(SetAppCompatStringPointer) \
    X(UpdateHMDEmulationStatus)

// exports this Windows' dxgi.dll doesn't have (some are Windows 11 only) answer "unsupported"
extern "C" HRESULT WINAPI fwd_missing() { return DXGI_ERROR_UNSUPPORTED; }
#define X(n) extern "C" { void* real_##n = (void*)fwd_missing; }
DXGI_EXPORTS(X)
#undef X
#define X(n) asm(".text\n.globl Fwd_" #n "\nFwd_" #n ":\n  jmp *real_" #n "(%rip)\n");
DXGI_EXPORTS(X)
#undef X

typedef HRESULT(WINAPI* PFN_CF)(REFIID, void**);
typedef HRESULT(WINAPI* PFN_CF2)(UINT, REFIID, void**);
static PFN_CF real_CreateDXGIFactory, real_CreateDXGIFactory1;
static PFN_CF2 real_CreateDXGIFactory2;
static HMODULE g_real_dxgi;

static void load_real() {
    wchar_t p[MAX_PATH];
    GetSystemDirectoryW(p, MAX_PATH);
    wcscat(p, L"\\dxgi.dll");
    g_real_dxgi = LoadLibraryW(p);
    if (!g_real_dxgi) return;
#define X(n) if (void* p = (void*)GetProcAddress(g_real_dxgi, #n)) real_##n = p;
    DXGI_EXPORTS(X)
#undef X
    real_CreateDXGIFactory = (PFN_CF)GetProcAddress(g_real_dxgi, "CreateDXGIFactory");
    real_CreateDXGIFactory1 = (PFN_CF)GetProcAddress(g_real_dxgi, "CreateDXGIFactory1");
    real_CreateDXGIFactory2 = (PFN_CF2)GetProcAddress(g_real_dxgi, "CreateDXGIFactory2");
}

// ------------------------------------------------------------------ logging / files
static std::wstring g_dir;
static FILE* g_log;
static SRWLOCK g_lock = SRWLOCK_INIT;   // guards files and maps below
static FILE* g_blobs;
static FILE* g_psos;
static std::unordered_set<uint64_t> g_blob_seen, g_pso_seen;
static std::unordered_map<void*, uint64_t> g_rootsig;   // live root signature -> blob hash
static std::atomic<long> g_recorded{0}, g_dupes{0}, g_skipped{0};
static bool g_cfg_replay = true, g_cfg_precompile = true;
static int g_cfg_replay_prio = THREAD_PRIORITY_NORMAL, g_cfg_pre_prio = THREAD_PRIORITY_NORMAL;
static int g_cfg_replay_threads = -2, g_cfg_pre_threads = -2;   // -2 = the CPU threads the game leaves idle

// the CPU threads the game can run on: its affinity mask, or with more than 64 CPU threads only its
// own processor group (before Windows 11 a process stays in one group)
static int usable_cpus() {
    int n = 0;
    if (GetActiveProcessorGroupCount() > 1) {
        GROUP_AFFINITY ga = {};
        if (GetThreadGroupAffinity(GetCurrentThread(), &ga)) n = __builtin_popcountll(ga.Mask);
    } else {
        DWORD_PTR proc = 0, sys = 0;
        if (GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys)) n = __builtin_popcountll(proc);
    }
    if (n < 1) n = (int)GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    return std::max(1, n);
}
// configured: -1 = every CPU thread, 0 = half the CPU threads, 1-64 = exactly that many;
// -2 (the CPU threads the game isn't using) is measured in start_threads and falls back to half
static int thread_count(int configured) {
    int all = usable_cpus();
    int n = configured > 0 ? configured : configured == -1 ? all : all / 2;
    return std::max(1, std::min(n, 64));   // 64: the most WaitForMultipleObjects takes
}
// this session's pipelines and root signatures, collected until the pre-compile has learned from them
static std::vector<std::vector<uint8_t>> g_live_gfx, g_live_cs;
static std::unordered_map<uint64_t, std::vector<uint8_t>> g_live_roots;
static bool g_live_taken = false;   // guarded by g_lock

static std::wstring g_log_path;
static SRWLOCK g_log_lock = SRWLOCK_INIT;
static int g_log_level = 3, g_log_progress = 500;
static long long g_log_max = 200LL << 20;
static bool g_exiting = false;   // at process exit a killed thread may still hold g_log_lock

static void vlog(const char* fmt, va_list a) {
    if (!g_log) return;
    if (g_exiting) { if (!TryAcquireSRWLockExclusive(&g_log_lock)) return; }
    else AcquireSRWLockExclusive(&g_log_lock);
    if (!g_log) { ReleaseSRWLockExclusive(&g_log_lock); return; }   // a failed rollover closed it meanwhile
    if (g_log_max > 0 && _ftelli64(g_log) > g_log_max) {   // roll over to wh_pso.old.log
        fclose(g_log);
        std::wstring old = g_log_path.substr(0, g_log_path.size() - 4) + L".old.log";
        bool moved = MoveFileExW(g_log_path.c_str(), old.c_str(), MOVEFILE_REPLACE_EXISTING);
        g_log = _wfopen(g_log_path.c_str(), moved ? L"a" : L"w");   // can't keep it: start it over
        if (!g_log) { ReleaseSRWLockExclusive(&g_log_lock); return; }
    }
    SYSTEMTIME t; GetLocalTime(&t);
    fprintf(g_log, "[%04d-%02d-%02d %02d:%02d:%02d.%03d] ", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond, t.wMilliseconds);
    vfprintf(g_log, fmt, a);
    fputc('\n', g_log); fflush(g_log);
    ReleaseSRWLockExclusive(&g_log_lock);
}
static void logf(const char* fmt, ...) { va_list a; va_start(a, fmt); vlog(fmt, a); va_end(a); }
// Level=2: every failure with its error codes and reasons
static void logd(const char* fmt, ...) {
    if (g_log_level < 2) return;
    va_list a; va_start(a, fmt); vlog(fmt, a); va_end(a);
}
// Level=3: everything (every pipeline built or created)
static void logv(const char* fmt, ...) {
    if (g_log_level < 3) return;
    va_list a; va_start(a, fmt); vlog(fmt, a); va_end(a);
}

static uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ULL) {
    const uint8_t* b = (const uint8_t*)p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

static const uint32_t MAGIC_BLOB = 0x424F4C42, MAGIC_PSO = 0x314F5350;   // "BLOB", "PSO1"
enum { BLOB_SHADER = 1, BLOB_ROOTSIG = 2 };
enum { PSO_GFX = 1, PSO_CS = 2, PSO_STREAM = 3 };

// the id of a shader or root signature in blobs.bin (and in the pre-compile's progress)
static uint64_t blob_key(uint32_t kind, const void* p, size_t n) {
    uint64_t h = fnv(p, n) ^ ((uint64_t)n << 40) ^ kind;
    return h ? h : 1;
}
// caller holds g_lock
static uint64_t put_blob(uint32_t kind, const void* p, size_t n) {
    if (!p || !n) return 0;
    uint64_t h = blob_key(kind, p, n);
    if (g_blobs && g_blob_seen.insert(h).second) {
        uint32_t hdr[2] = {MAGIC_BLOB, kind}; uint32_t sz = (uint32_t)n;
        fwrite(hdr, 4, 2, g_blobs); fwrite(&h, 8, 1, g_blobs); fwrite(&sz, 4, 1, g_blobs);
        fwrite(p, 1, n, g_blobs); fflush(g_blobs);
    }
    return h;
}

struct Buf {
    std::vector<uint8_t> v;
    void raw(const void* p, size_t n) { v.insert(v.end(), (const uint8_t*)p, (const uint8_t*)p + n); }
    void u32(uint32_t x) { raw(&x, 4); }
    void u64(uint64_t x) { raw(&x, 8); }
};

// caller holds g_lock
// returns true when this pipeline was never recorded before (on any launch)
static bool put_pso(uint32_t kind, const Buf& b) {
    uint64_t h = fnv(b.v.data(), b.v.size()) ^ kind;
    if (!g_pso_seen.insert(h).second) { g_dupes++; return false; }
    if (!g_psos) return true;
    uint32_t hdr[3] = {MAGIC_PSO, kind, (uint32_t)b.v.size()};
    fwrite(hdr, 4, 3, g_psos); fwrite(b.v.data(), 1, b.v.size(), g_psos); fflush(g_psos);
    g_recorded++;
    return true;
}

static void ser_shader(Buf& b, const D3D12_SHADER_BYTECODE& s) { b.u64(put_blob(BLOB_SHADER, s.pShaderBytecode, s.BytecodeLength)); }
static void ser_root(Buf& b, void* rs) {
    if (!rs) { b.u64(0); return; }
    auto it = g_rootsig.find(rs);
    b.u64(it == g_rootsig.end() ? ~0ULL : it->second);
}
static void ser_layout(Buf& b, const D3D12_INPUT_ELEMENT_DESC* e, UINT n) {
    b.u32(e ? n : 0);
    for (UINT i = 0; e && i < n; i++) {
        uint32_t l = e[i].SemanticName ? (uint32_t)strlen(e[i].SemanticName) : 0;
        b.u32(l); b.raw(e[i].SemanticName, l);
        b.u32(e[i].SemanticIndex); b.u32(e[i].Format); b.u32(e[i].InputSlot);
        b.u32(e[i].AlignedByteOffset); b.u32(e[i].InputSlotClass); b.u32(e[i].InstanceDataStepRate);
    }
}
static void ser_so(Buf& b, const D3D12_STREAM_OUTPUT_DESC& so) {
    b.u32(so.pSODeclaration ? so.NumEntries : 0);
    for (UINT i = 0; so.pSODeclaration && i < so.NumEntries; i++) {
        const D3D12_SO_DECLARATION_ENTRY& e = so.pSODeclaration[i];
        uint32_t l = e.SemanticName ? (uint32_t)strlen(e.SemanticName) : 0;
        b.u32(e.Stream); b.u32(l); b.raw(e.SemanticName, l); b.u32(e.SemanticIndex);
        b.u32(e.StartComponent); b.u32(e.ComponentCount); b.u32(e.OutputSlot);
    }
    b.u32(so.pBufferStrides ? so.NumStrides : 0);
    for (UINT i = 0; so.pBufferStrides && i < so.NumStrides; i++) b.u32(so.pBufferStrides[i]);
    b.u32(so.RasterizedStream);
}

// ------------------------------------------------------------------ pipeline stream layout
// Subobject types and inner sizes (bytes) / alignment, from d3d12.h (Agility SDK 1.6xx).
enum {
    SO_ROOT = 0, SO_VS, SO_PS, SO_DS, SO_HS, SO_GS, SO_CS, SO_STREAM_OUTPUT, SO_BLEND, SO_SAMPLE_MASK,
    SO_RAST, SO_DEPTH, SO_INPUT_LAYOUT, SO_STRIP, SO_TOPO, SO_RTV, SO_DSV, SO_SAMPLE_DESC, SO_NODE,
    SO_CACHED, SO_FLAGS, SO_DEPTH1, SO_VIEW_INST, SO_AS = 24, SO_MS, SO_DEPTH2, SO_RAST1, SO_RAST2
};
static bool so_shape(uint32_t t, size_t* size, size_t* align) {
    switch (t) {
    case SO_ROOT: *size = 8; *align = 8; return true;
    case SO_VS: case SO_PS: case SO_DS: case SO_HS: case SO_GS: case SO_CS: case SO_AS: case SO_MS:
    case SO_CACHED: *size = 16; *align = 8; return true;
    case SO_STREAM_OUTPUT: *size = 32; *align = 8; return true;
    case SO_BLEND: *size = 328; *align = 4; return true;
    case SO_SAMPLE_MASK: case SO_STRIP: case SO_TOPO: case SO_DSV: case SO_NODE: case SO_FLAGS:
        *size = 4; *align = 4; return true;
    case SO_RAST: case SO_RAST1: *size = 44; *align = 4; return true;
    case SO_RAST2: *size = 40; *align = 4; return true;
    case SO_DEPTH: *size = 52; *align = 4; return true;
    case SO_DEPTH1: *size = 56; *align = 4; return true;
    case SO_DEPTH2: *size = 60; *align = 4; return true;
    case SO_INPUT_LAYOUT: *size = 16; *align = 8; return true;
    case SO_RTV: *size = 36; *align = 4; return true;
    case SO_SAMPLE_DESC: *size = 8; *align = 4; return true;
    case SO_VIEW_INST: *size = 24; *align = 8; return true;
    }
    return false;
}
static size_t up(size_t x, size_t a) { return (x + a - 1) & ~(a - 1); }

struct VIEW_LOC { UINT vp, rt; };
struct VIEW_INST { UINT n; const VIEW_LOC* p; UINT flags; };

// caller holds g_lock. Returns false if the stream has a subobject type we don't know.
static bool ser_stream(Buf& b, const void* data, size_t size) {
    const uint8_t* p = (const uint8_t*)data; size_t off = 0; uint32_t count = 0;
    Buf body;
    while (off < size) {
        uint32_t t = *(const uint32_t*)(p + off);
        size_t sz, al;
        if (!so_shape(t, &sz, &al)) { logf("unknown pipeline subobject type %u, not recorded", t); return false; }
        const uint8_t* in = p + off + up(4, al);
        if (t != SO_CACHED) {
            body.u32(t); count++;
            switch (t) {
            case SO_ROOT: ser_root(body, *(void* const*)in); break;
            case SO_VS: case SO_PS: case SO_DS: case SO_HS: case SO_GS: case SO_CS: case SO_AS: case SO_MS:
                ser_shader(body, *(const D3D12_SHADER_BYTECODE*)in); break;
            case SO_STREAM_OUTPUT: ser_so(body, *(const D3D12_STREAM_OUTPUT_DESC*)in); break;
            case SO_INPUT_LAYOUT: { auto* l = (const D3D12_INPUT_LAYOUT_DESC*)in; ser_layout(body, l->pInputElementDescs, l->NumElements); break; }
            case SO_VIEW_INST: { auto* v = (const VIEW_INST*)in; body.u32(v->p ? v->n : 0); if (v->p) body.raw(v->p, v->n * sizeof(VIEW_LOC)); body.u32(v->flags); break; }
            default: body.raw(in, sz);
            }
        }
        off += up(up(4, al) + sz, 8);
    }
    b.u32(count); b.raw(body.v.data(), body.v.size());
    return true;
}

// ------------------------------------------------------------------ device hooks
typedef HRESULT(STDMETHODCALLTYPE* PFN_RS)(ID3D12Device*, UINT, const void*, SIZE_T, REFIID, void**);
typedef HRESULT(STDMETHODCALLTYPE* PFN_GFX)(ID3D12Device*, const D3D12_GRAPHICS_PIPELINE_STATE_DESC*, REFIID, void**);
typedef HRESULT(STDMETHODCALLTYPE* PFN_CS)(ID3D12Device*, const D3D12_COMPUTE_PIPELINE_STATE_DESC*, REFIID, void**);
struct STREAM_DESC { SIZE_T size; void* p; };
typedef HRESULT(STDMETHODCALLTYPE* PFN_STREAM)(ID3D12Device*, const STREAM_DESC*, REFIID, void**);
static PFN_RS o_CreateRootSignature;
static PFN_GFX o_CreateGraphicsPipelineState;
static PFN_CS o_CreateComputePipelineState;
static PFN_STREAM o_CreatePipelineState;
static std::atomic<int> g_replay_started{0};
static void start_replay(ID3D12Device* dev);

// Logging only: how long the game waits on each pipeline it creates. A driver cache hit takes well
// under a millisecond; a real compile takes several to hundreds. [0] = pipelines recorded on an
// earlier launch (Replay covers these), [1] = never seen before (only the pre-compile covers these).
static const double T_EDGE[] = {0.5, 2, 10, 50};
static std::atomic<long> g_t_hist[2][5];
static std::atomic<long long> g_t_slow_us[2];
static double qpc_ms() {
    static LARGE_INTEGER f = [] { LARGE_INTEGER x; QueryPerformanceFrequency(&x); return x; }();
    LARGE_INTEGER c; QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
}
static std::atomic<int> g_pre_finished{0};
static std::atomic<int> g_new_lines{0};
// Logging only: where the pre-compile found each shader ("EXE", the file it was in, or the archive family)
static SRWLOCK g_fam_lock = SRWLOCK_INIT;
static std::unordered_map<uint64_t, std::string> g_shader_family;
static void note_family(uint64_t k, const std::string& fam) {
    if (!g_log) return;
    AcquireSRWLockExclusive(&g_fam_lock); g_shader_family.emplace(k, fam); ReleaseSRWLockExclusive(&g_fam_lock);
}
static void describe_shader(char* out, size_t n, const char* stage, uint64_t k) {
    AcquireSRWLockShared(&g_fam_lock);
    auto f = g_shader_family.find(k);
    std::string fam = f == g_shader_family.end() ? std::string() : f->second;
    ReleaseSRWLockShared(&g_fam_lock);
    snprintf(out, n, " %s %016llx [%s]", stage, (unsigned long long)k, fam.empty() ? "not found in the game files" : fam.c_str());
}
// keys: the pipeline's serialized record; graphics = root, VS, PS, DS, HS, GS (u64 each), compute = root, CS
static void time_pso(bool fresh, double ms, const char* what, const std::vector<uint8_t>* rec) {
    if (!g_log) return;
    int b = 0; while (b < 4 && ms >= T_EDGE[b]) b++;
    g_t_hist[fresh][b]++;
    if (ms >= 2) g_t_slow_us[fresh] += (long long)(ms * 1000);
    bool report = fresh && g_pre_finished;
    if (!report && ms < 50 && g_log_level < 3) return;
    if (report && g_log_level < 2) return;
    char ids[600] = ""; size_t o = 0;
    if (rec && rec->size() >= 48 && strcmp(what, "graphics") == 0) {
        uint64_t k[6]; memcpy(k, rec->data(), 48);
        static const char* nm[6] = {"", "VS", "PS", "DS", "HS", "GS"};
        for (int i = 1; i < 6 && o < sizeof ids - 1; i++) if (k[i]) { describe_shader(ids + o, sizeof ids - o, nm[i], k[i]); o = strlen(ids); }
    } else if (rec && rec->size() >= 16) {
        uint64_t k[2]; memcpy(k, rec->data(), 16);
        describe_shader(ids, sizeof ids, "CS", k[1]);
    }
    // after the pre-compile finished, every new pipeline is logged, cache hit or not
    if (report && g_new_lines++ < 5000)
        logf("new pipeline after pre-compile: %s, %.2f ms, %s:%s", what, ms, ms >= 2 ? "COMPILED ON THE SPOT" : "from cache", ids);
    else if (!report && ms >= 50)
        logd("game waited %.0f ms for a %s %s pipeline:%s", ms, fresh ? "new" : "repeat", what, ids);
    else if (!report)
        logv("game pipeline: %s %s, %.2f ms%s:%s", fresh ? "new" : "repeat", what, ms, ms >= 2 ? " (compiled)" : "", ids);
}
static void log_pso_times(const char* when) {
    for (int f = 0; f < 2; f++) {
        long n = 0; for (int b = 0; b < 5; b++) n += g_t_hist[f][b].load();
        if (!n) continue;
        long slow = g_t_hist[f][2] + g_t_hist[f][3] + g_t_hist[f][4];
        logf("game pipelines %s, %s: %ld created; <0.5ms %ld, 0.5-2ms %ld, 2-10ms %ld, 10-50ms %ld, >50ms %ld; "
             "%ld (%.0f%%) were compiled on the spot (>=2ms), %.1fs spent waiting on them",
             when, f ? "new (never created before)" : "repeats (created before, this or an earlier launch)", n, g_t_hist[f][0].load(), g_t_hist[f][1].load(), g_t_hist[f][2].load(),
             g_t_hist[f][3].load(), g_t_hist[f][4].load(), slow, 100.0 * slow / n, g_t_slow_us[f].load() / 1e6);
        for (int b = 0; b < 5; b++) g_t_hist[f][b] = 0;
        g_t_slow_us[f] = 0;
    }
}

static HRESULT STDMETHODCALLTYPE h_CreateRootSignature(ID3D12Device* d, UINT node, const void* blob, SIZE_T n, REFIID riid, void** out) {
    HRESULT hr = o_CreateRootSignature(d, node, blob, n, riid, out);
    if (SUCCEEDED(hr) && out && *out) {
        AcquireSRWLockExclusive(&g_lock);
        uint64_t h = put_blob(BLOB_ROOTSIG, blob, n);
        g_rootsig[*out] = h;
        if (h && !g_live_taken && !g_live_roots.count(h)) g_live_roots[h].assign((const uint8_t*)blob, (const uint8_t*)blob + n);
        ReleaseSRWLockExclusive(&g_lock);
    }
    if (!g_replay_started.exchange(1)) start_replay(d);
    return hr;
}

static HRESULT STDMETHODCALLTYPE h_CreateGraphicsPipelineState(ID3D12Device* d, const D3D12_GRAPHICS_PIPELINE_STATE_DESC* s, REFIID riid, void** out) {
    double t0 = qpc_ms();
    HRESULT hr = o_CreateGraphicsPipelineState(d, s, riid, out);
    double ms = qpc_ms() - t0;
    if (SUCCEEDED(hr) && s) {
        Buf b;
        AcquireSRWLockExclusive(&g_lock);
        ser_root(b, s->pRootSignature);
        ser_shader(b, s->VS); ser_shader(b, s->PS); ser_shader(b, s->DS); ser_shader(b, s->HS); ser_shader(b, s->GS);
        ser_so(b, s->StreamOutput);
        b.raw(&s->BlendState, sizeof s->BlendState); b.u32(s->SampleMask);
        b.raw(&s->RasterizerState, sizeof s->RasterizerState); b.raw(&s->DepthStencilState, sizeof s->DepthStencilState);
        ser_layout(b, s->InputLayout.pInputElementDescs, s->InputLayout.NumElements);
        b.u32(s->IBStripCutValue); b.u32(s->PrimitiveTopologyType); b.u32(s->NumRenderTargets);
        b.raw(s->RTVFormats, sizeof s->RTVFormats); b.u32(s->DSVFormat);
        b.raw(&s->SampleDesc, sizeof s->SampleDesc); b.u32(s->NodeMask); b.u32(s->Flags);
        bool fresh = put_pso(PSO_GFX, b);
        if (!g_live_taken) g_live_gfx.push_back(b.v);
        ReleaseSRWLockExclusive(&g_lock);
        time_pso(fresh, ms, "graphics", &b.v);
    }
    return hr;
}

static HRESULT STDMETHODCALLTYPE h_CreateComputePipelineState(ID3D12Device* d, const D3D12_COMPUTE_PIPELINE_STATE_DESC* s, REFIID riid, void** out) {
    double t0 = qpc_ms();
    HRESULT hr = o_CreateComputePipelineState(d, s, riid, out);
    double ms = qpc_ms() - t0;
    if (SUCCEEDED(hr) && s) {
        Buf b;
        AcquireSRWLockExclusive(&g_lock);
        ser_root(b, s->pRootSignature); ser_shader(b, s->CS); b.u32(s->NodeMask); b.u32(s->Flags);
        bool fresh = put_pso(PSO_CS, b);
        if (!g_live_taken) g_live_cs.push_back(b.v);
        ReleaseSRWLockExclusive(&g_lock);
        time_pso(fresh, ms, "compute", &b.v);
    }
    return hr;
}

static HRESULT STDMETHODCALLTYPE h_CreatePipelineState(ID3D12Device* d, const STREAM_DESC* s, REFIID riid, void** out) {
    double t0 = qpc_ms();
    HRESULT hr = o_CreatePipelineState(d, s, riid, out);
    double ms = qpc_ms() - t0;
    if (SUCCEEDED(hr) && s && s->p) {
        Buf b; bool fresh = false, ok;
        AcquireSRWLockExclusive(&g_lock);
        if ((ok = ser_stream(b, s->p, s->size))) fresh = put_pso(PSO_STREAM, b); else g_skipped++;
        ReleaseSRWLockExclusive(&g_lock);
        if (ok) time_pso(fresh, ms, "stream", nullptr);
    }
    return hr;
}

static void patch(void** vt, int i, void* fn, void** orig) {
    DWORD old;
    VirtualProtect(&vt[i], sizeof(void*), PAGE_READWRITE, &old);
    *orig = vt[i]; vt[i] = fn;
    VirtualProtect(&vt[i], sizeof(void*), old, &old);
}

static void open_files();
static std::atomic<int> g_hook_state{0};   // 0 = not yet, 1 = installing, 2 = done
static void install_hooks() {
    int z = 0;
    if (!g_hook_state.compare_exchange_strong(z, 1)) return;
    open_files();   // here rather than in DllMain: it reads the existing recordings
    HMODULE d3d = LoadLibraryW(L"d3d12.dll");
    auto create = d3d ? (PFN_D3D12_CREATE_DEVICE)GetProcAddress(d3d, "D3D12CreateDevice") : nullptr;
    // a throwaway device, only to find the vtable. It belongs to the D3D12 runtime and is the same
    // for every adapter, so if the default adapter can't make one, any other adapter or WARP will do.
    ID3D12Device* dev = nullptr;
    if (create && FAILED(create(nullptr, D3D_FEATURE_LEVEL_11_0, IID_ID3D12Device, (void**)&dev))) dev = nullptr;
    IDXGIFactory4* fac = nullptr;
    if (!dev && create && real_CreateDXGIFactory1 && SUCCEEDED(real_CreateDXGIFactory1(IID_IDXGIFactory4, (void**)&fac)) && fac) {
        IDXGIAdapter1* a = nullptr;
        for (UINT i = 0; !dev && fac->EnumAdapters1(i, &a) == S_OK; i++) {
            if (FAILED(create(a, D3D_FEATURE_LEVEL_11_0, IID_ID3D12Device, (void**)&dev))) dev = nullptr;
            a->Release();
        }
        IDXGIAdapter* w = nullptr;
        if (!dev && SUCCEEDED(fac->EnumWarpAdapter(IID_IDXGIAdapter, (void**)&w)) && w) {
            if (FAILED(create(w, D3D_FEATURE_LEVEL_11_0, IID_ID3D12Device, (void**)&dev))) dev = nullptr;
            w->Release();
        }
        fac->Release();
    }
    if (dev) {
        void** vt = *(void***)dev;
        // ID3D12Device vtable: 10 CreateGraphicsPipelineState, 11 CreateComputePipelineState,
        // 16 CreateRootSignature; ID3D12Device2: 47 CreatePipelineState
        patch(vt, 10, (void*)h_CreateGraphicsPipelineState, (void**)&o_CreateGraphicsPipelineState);
        patch(vt, 11, (void*)h_CreateComputePipelineState, (void**)&o_CreateComputePipelineState);
        patch(vt, 16, (void*)h_CreateRootSignature, (void**)&o_CreateRootSignature);
        ID3D12Device* dev2 = nullptr;
        static const GUID IID_Dev2 = {0x30baa41e, 0xb15b, 0x475c, {0xa0, 0xbb, 0x1a, 0xf5, 0xc5, 0xb6, 0x43, 0x28}};
        if (SUCCEEDED(dev->QueryInterface(IID_Dev2, (void**)&dev2)) && dev2) {
            patch(*(void***)dev2, 47, (void*)h_CreatePipelineState, (void**)&o_CreatePipelineState);
            dev2->Release();
        }
        dev->Release();
        logf("hooks installed (stream PSO hook: %s)", o_CreatePipelineState ? "yes" : "no");
    } else {
        logf("could not create a D3D12 device on any adapter to install hooks; disabled");
    }
    g_hook_state = 2;
}

#include "overlay.inc"
#include "ini_default.h"

// ------------------------------------------------------------------ replay
struct Reader {
    const uint8_t* p; size_t n, o = 0; bool bad = false;
    bool need(size_t k) { if (o + k > n) bad = true; return !bad; }
    uint32_t u32() { if (!need(4)) return 0; uint32_t x; memcpy(&x, p + o, 4); o += 4; return x; }
    uint64_t u64() { if (!need(8)) return 0; uint64_t x; memcpy(&x, p + o, 8); o += 8; return x; }
    const uint8_t* raw(size_t k) { if (!need(k)) return nullptr; const uint8_t* r = p + o; o += k; return r; }
};

static std::vector<uint8_t> read_file(const std::wstring& path) {
    std::vector<uint8_t> v;
    FILE* f = _wfopen(path.c_str(), L"rb");
    if (!f) return v;
    fseek(f, 0, SEEK_END); long long n = _ftelli64(f); fseek(f, 0, SEEK_SET);
    if (n > 0) { v.resize((size_t)n); v.resize(fread(v.data(), 1, (size_t)n, f)); }
    fclose(f);
    return v;
}

// Which CPU threads is the game leaving idle? Busy time per logical processor, sampled over
// half a second. Returns how many are idle (under 30% busy) and their affinity mask. Only CPUs
// this process may run on count. -1 when it can't be measured.
static int idle_cpus(DWORD_PTR* mask) {
    typedef LONG(WINAPI * PFN_NTQSI)(ULONG, void*, ULONG, ULONG*);
    struct SPPI { LARGE_INTEGER idle, kernel, user, r1[2]; ULONG r2; };
    static PFN_NTQSI q = (PFN_NTQSI)GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQuerySystemInformation");
    *mask = 0;
    if (!q || GetActiveProcessorGroupCount() > 1) return -1;   // more than 64 CPU threads: masks don't cover them all
    DWORD_PTR proc = 0, sys = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys) || !proc) return -1;
    int np = (int)GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (np < 1) return -1;
    std::vector<SPPI> a(np), b(np); ULONG len = 0;
    if (q(8 /* SystemProcessorPerformanceInformation */, a.data(), (ULONG)(sizeof(SPPI) * np), &len) < 0) return -1;
    Sleep(500);
    if (q(8, b.data(), (ULONG)(sizeof(SPPI) * np), &len) < 0) return -1;
    int got = (int)(len / sizeof(SPPI)), n = 0;
    for (int i = 0; i < got && i < np; i++) {
        if (!((proc >> i) & 1)) continue;
        long long total = (b[i].kernel.QuadPart - a[i].kernel.QuadPart) + (b[i].user.QuadPart - a[i].user.QuadPart);   // kernel includes idle
        long long idle = b[i].idle.QuadPart - a[i].idle.QuadPart;
        double busy = total > 0 ? 1.0 - (double)idle / (double)total : 0.0;
        if (busy < 0.30) { *mask |= (DWORD_PTR)1 << i; n++; }
    }
    return n;
}

// start the worker threads for one job. With configured = -2 they run on the CPU threads the game
// isn't using (half of them when it uses all), and leave the game at least a quarter (2 at minimum).
static int start_threads(int configured, LPTHREAD_START_ROUTINE fn, void* arg, std::vector<HANDLE>& th, const char* job) {
    DWORD_PTR mask = 0; int n;
    if (configured == -2) {
        int all = usable_cpus(), idle = idle_cpus(&mask);
        int most = std::max(1, all - std::max(2, all / 4));
        if (idle >= 1) {
            n = std::min(idle, most);
            logf("%s: %d of %d CPU threads are idle; using %d of them", job, idle, all, n);
        } else {
            mask = 0; n = std::min(thread_count(-2), most);
            logf("%s: the game is using every CPU thread%s; using %d", job, idle < 0 ? " (or usage couldn't be read)" : "", n);
        }
    } else n = thread_count(configured);
    bool pin_failed = false;
    for (int i = 0; i < n; i++) {
        HANDLE h = CreateThread(nullptr, 0, fn, arg, CREATE_SUSPENDED, nullptr);
        if (!h) continue;
        if (mask && !SetThreadAffinityMask(h, mask)) pin_failed = true;
        ResumeThread(h);
        th.push_back(h);
    }
    if (pin_failed) logf("%s: could not keep the threads on the idle CPU threads", job);
    return (int)th.size();
}

struct Replay {
    ID3D12Device* dev;
    std::unordered_map<uint64_t, std::pair<const uint8_t*, uint32_t>> blobs;
    std::unordered_map<uint64_t, ID3D12RootSignature*> roots;
    std::vector<std::pair<uint32_t, std::pair<const uint8_t*, uint32_t>>> psos;
    std::vector<std::pair<uint32_t, std::pair<const uint8_t*, uint32_t>>> seed_psos;   // WH_SHADERS\seed
    std::vector<std::pair<uint32_t, std::pair<const uint8_t*, uint32_t>>> todo;        // what the replay rebuilds
    size_t todo_own = 0;                                                               // the first todo_own are this PC's
    std::vector<std::vector<uint8_t>> files;
    // The seed ships no shaders, only which ones its pipelines use: the pre-compile picks them up
    // from the game's own files (and records them, so later launches replay these at startup).
    std::unordered_set<uint64_t> wanted;   // seed shaders not recorded on this PC
    std::vector<size_t> missed;            // seed pipelines (todo indices) the replay had to skip for now
    long adopted = 0;                      // wanted shaders found this launch
    std::atomic<size_t> next{0};
    std::atomic<long> ok{0}, failed{0}, missing{0};
    long base = 0;                         // ok + failed + missing before this run (the overlay counts from here)
    std::atomic<long> own_ok{0}, own_slow{0};   // this PC's recordings rebuilt, and how many weren't in the driver cache
    SRWLOCK lock = SRWLOCK_INIT;
};
static Replay* g_rp;

static D3D12_SHADER_BYTECODE rd_shader(Replay* r, Reader& in, bool* miss) {
    uint64_t h = in.u64();
    D3D12_SHADER_BYTECODE s = {nullptr, 0};
    if (!h) return s;
    auto it = r->blobs.find(h);
    if (it == r->blobs.end()) { *miss = true; return s; }
    s.pShaderBytecode = it->second.first; s.BytecodeLength = it->second.second;
    return s;
}
// the root signature with this id, created once from a recorded blob (or one in extra)
static ID3D12RootSignature* root_for(Replay* r, uint64_t h, const std::unordered_map<uint64_t, std::vector<uint8_t>>* extra = nullptr) {
    AcquireSRWLockExclusive(&r->lock);
    ID3D12RootSignature* rs = nullptr;
    auto it = r->roots.find(h);
    if (it != r->roots.end()) rs = it->second;
    else {
        auto d = r->blobs.find(h);
        if (d != r->blobs.end())
            o_CreateRootSignature(r->dev, 0, d->second.first, d->second.second, IID_ID3D12RootSignature, (void**)&rs);
        else if (extra) {
            auto b = extra->find(h);
            if (b != extra->end()) o_CreateRootSignature(r->dev, 0, b->second.data(), b->second.size(), IID_ID3D12RootSignature, (void**)&rs);
        }
        r->roots[h] = rs;
    }
    ReleaseSRWLockExclusive(&r->lock);
    return rs;
}
static ID3D12RootSignature* rd_root(Replay* r, Reader& in, bool* miss) {
    uint64_t h = in.u64();
    if (!h) return nullptr;
    ID3D12RootSignature* rs = h == ~0ULL ? nullptr : root_for(r, h);
    if (!rs) *miss = true;
    return rs;
}
// strings and arrays referenced by a rebuilt description live here until it is used
struct Keep {
    std::vector<std::string> names;
    std::vector<std::vector<D3D12_INPUT_ELEMENT_DESC>> layouts;
    std::vector<std::vector<D3D12_SO_DECLARATION_ENTRY>> sos;
    std::vector<std::vector<UINT>> strides;
    std::vector<std::vector<VIEW_LOC>> views;
    Keep() { names.reserve(512); }
};
static D3D12_INPUT_LAYOUT_DESC rd_layout(Reader& in, Keep& k) {
    uint32_t n = in.u32();
    if (n > 32) { in.bad = true; n = 0; }   // D3D12 allows 32 input elements: a damaged record
    k.layouts.emplace_back(n);
    auto& v = k.layouts.back();
    for (uint32_t i = 0; i < n && !in.bad; i++) {
        uint32_t l = in.u32(); const uint8_t* s = in.raw(l);
        k.names.emplace_back(s ? std::string((const char*)s, l) : std::string());
        v[i].SemanticName = k.names.back().c_str();
        v[i].SemanticIndex = in.u32(); v[i].Format = (DXGI_FORMAT)in.u32(); v[i].InputSlot = in.u32();
        v[i].AlignedByteOffset = in.u32(); v[i].InputSlotClass = (D3D12_INPUT_CLASSIFICATION)in.u32();
        v[i].InstanceDataStepRate = in.u32();
    }
    return {n ? v.data() : nullptr, n};
}
static D3D12_STREAM_OUTPUT_DESC rd_so(Reader& in, Keep& k) {
    D3D12_STREAM_OUTPUT_DESC so = {};
    uint32_t n = in.u32();
    if (n > 512) { in.bad = true; n = 0; }
    k.sos.emplace_back(n);
    auto& v = k.sos.back();
    for (uint32_t i = 0; i < n && !in.bad; i++) {
        v[i].Stream = in.u32();
        uint32_t l = in.u32(); const uint8_t* s = in.raw(l);
        k.names.emplace_back(s ? std::string((const char*)s, l) : std::string());
        v[i].SemanticName = k.names.back().c_str();
        v[i].SemanticIndex = in.u32(); v[i].StartComponent = (BYTE)in.u32();
        v[i].ComponentCount = (BYTE)in.u32(); v[i].OutputSlot = (BYTE)in.u32();
    }
    uint32_t ns = in.u32();
    if (ns > 4) { in.bad = true; ns = 0; }
    k.strides.emplace_back(ns);
    for (uint32_t i = 0; i < ns && !in.bad; i++) k.strides.back()[i] = in.u32();
    so.pSODeclaration = n ? v.data() : nullptr; so.NumEntries = n;
    so.pBufferStrides = ns ? k.strides.back().data() : nullptr; so.NumStrides = ns;
    so.RasterizedStream = in.u32();
    return so;
}

static HRESULT replay_one(Replay* r, uint32_t kind, Reader& in) {
    bool miss = false; Keep k; ID3D12PipelineState* pso = nullptr; HRESULT hr = E_FAIL;
    if (kind == PSO_GFX) {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC d = {};
        d.pRootSignature = rd_root(r, in, &miss);
        d.VS = rd_shader(r, in, &miss); d.PS = rd_shader(r, in, &miss); d.DS = rd_shader(r, in, &miss);
        d.HS = rd_shader(r, in, &miss); d.GS = rd_shader(r, in, &miss);
        d.StreamOutput = rd_so(in, k);
        if (auto p = in.raw(sizeof d.BlendState)) memcpy(&d.BlendState, p, sizeof d.BlendState);
        d.SampleMask = in.u32();
        if (auto p = in.raw(sizeof d.RasterizerState)) memcpy(&d.RasterizerState, p, sizeof d.RasterizerState);
        if (auto p = in.raw(sizeof d.DepthStencilState)) memcpy(&d.DepthStencilState, p, sizeof d.DepthStencilState);
        d.InputLayout = rd_layout(in, k);
        d.IBStripCutValue = (D3D12_INDEX_BUFFER_STRIP_CUT_VALUE)in.u32();
        d.PrimitiveTopologyType = (D3D12_PRIMITIVE_TOPOLOGY_TYPE)in.u32();
        d.NumRenderTargets = in.u32();
        if (auto p = in.raw(sizeof d.RTVFormats)) memcpy(d.RTVFormats, p, sizeof d.RTVFormats);
        d.DSVFormat = (DXGI_FORMAT)in.u32();
        if (auto p = in.raw(sizeof d.SampleDesc)) memcpy(&d.SampleDesc, p, sizeof d.SampleDesc);
        d.NodeMask = in.u32(); d.Flags = (D3D12_PIPELINE_STATE_FLAGS)in.u32();
        if (miss || in.bad) return S_FALSE;
        hr = o_CreateGraphicsPipelineState(r->dev, &d, IID_ID3D12PipelineState, (void**)&pso);
    } else if (kind == PSO_CS) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC d = {};
        d.pRootSignature = rd_root(r, in, &miss); d.CS = rd_shader(r, in, &miss);
        d.NodeMask = in.u32(); d.Flags = (D3D12_PIPELINE_STATE_FLAGS)in.u32();
        if (miss || in.bad) return S_FALSE;
        hr = o_CreateComputePipelineState(r->dev, &d, IID_ID3D12PipelineState, (void**)&pso);
    } else if (kind == PSO_STREAM) {
        if (!o_CreatePipelineState) return S_FALSE;
        std::vector<uint8_t> st; st.reserve(1024);
        uint32_t n = in.u32();
        for (uint32_t i = 0; i < n && !in.bad; i++) {
            uint32_t t = in.u32(); size_t sz, al;
            if (!so_shape(t, &sz, &al)) return S_FALSE;
            size_t base = st.size(), io = up(4, al);
            st.resize(base + up(io + sz, 8), 0);
            memcpy(&st[base], &t, 4);
            uint8_t inner[328] = {};
            switch (t) {
            case SO_ROOT: { void* rs = rd_root(r, in, &miss); memcpy(inner, &rs, 8); break; }
            case SO_VS: case SO_PS: case SO_DS: case SO_HS: case SO_GS: case SO_CS: case SO_AS: case SO_MS: {
                D3D12_SHADER_BYTECODE s = rd_shader(r, in, &miss); memcpy(inner, &s, 16); break; }
            case SO_STREAM_OUTPUT: { D3D12_STREAM_OUTPUT_DESC so = rd_so(in, k); memcpy(inner, &so, sizeof so); break; }
            case SO_INPUT_LAYOUT: { D3D12_INPUT_LAYOUT_DESC l = rd_layout(in, k); memcpy(inner, &l, sizeof l); break; }
            case SO_VIEW_INST: {
                VIEW_INST v = {}; v.n = in.u32();
                if (v.n > 4) { in.bad = true; v.n = 0; }
                k.views.emplace_back(v.n);
                if (v.n) { if (auto p = in.raw(v.n * sizeof(VIEW_LOC))) memcpy(k.views.back().data(), p, v.n * sizeof(VIEW_LOC)); v.p = k.views.back().data(); }
                v.flags = in.u32(); memcpy(inner, &v, sizeof v); break; }
            default: if (auto p = in.raw(sz)) memcpy(inner, p, sz);
            }
            memcpy(&st[base + io], inner, sz);
        }
        if (miss || in.bad) return S_FALSE;
        STREAM_DESC sd = {st.size(), st.data()};
        hr = o_CreatePipelineState(r->dev, &sd, IID_ID3D12PipelineState, (void**)&pso);
    }
    if (pso) pso->Release();
    return hr;
}

static DWORD WINAPI replay_worker(void* arg) {
    Replay* r = (Replay*)arg;
    SetThreadPriority(GetCurrentThread(), g_cfg_replay_prio);
    for (;;) {
        size_t i = r->next++;
        if (i >= r->todo.size() || r->dev->GetDeviceRemovedReason() != S_OK) break;
        Reader in{r->todo[i].second.first, r->todo[i].second.second};
        double t0 = qpc_ms();
        HRESULT hr;
        try { hr = replay_one(r, r->todo[i].first, in); } catch (...) { hr = E_OUTOFMEMORY; }
        if (hr == S_OK && i < r->todo_own) {   // recorded here before, so normally a driver cache hit
            r->own_ok++;
            if (qpc_ms() - t0 >= 5) r->own_slow++;
        }
        if (hr == S_FALSE && i >= r->todo_own) { AcquireSRWLockExclusive(&r->lock); r->missed.push_back(i); ReleaseSRWLockExclusive(&r->lock); }
        if (hr == S_FALSE) r->missing++;
        else if (SUCCEEDED(hr)) r->ok++;
        else { r->failed++; logd("replay: pipeline %zu failed, hr=0x%08lx", i, (unsigned long)hr); }
        ov_set(OV_REPLAY, r->ok + r->failed + r->missing - r->base, (long)r->todo.size());
    }
    return 0;
}

static void run_replay(Replay* r, const char* job) {
    r->next = 0; r->base = r->ok + r->failed + r->missing;
    std::vector<HANDLE> th;
    int n = start_threads(g_cfg_replay_threads, replay_worker, r, th, job);
    logf("%s: rebuilding %zu pipelines (%zu recorded on this PC) on %d background threads", job, r->todo.size(), r->todo_own, n);
    if (!th.empty()) WaitForMultipleObjects((DWORD)th.size(), th.data(), TRUE, INFINITE);
    for (HANDLE h : th) CloseHandle(h);
}

static std::wstring g_seed_dir;
static void load_records(Replay* r, const std::wstring& name, bool blobs_file, bool seed = false) {
    r->files.push_back(read_file((seed ? g_seed_dir : g_dir) + name));
    const std::vector<uint8_t>& f = r->files.back();
    Reader in{f.data(), f.size()};
    while (in.o + 12 <= in.n) {
        uint32_t magic = in.u32();
        if (blobs_file && magic == MAGIC_BLOB) {
            in.u32(); uint64_t h = in.u64(); uint32_t sz = in.u32(); const uint8_t* p = in.raw(sz);
            if (!p) break;
            if (seed) r->blobs.emplace(h, std::make_pair(p, sz)); else r->blobs[h] = {p, sz};
        } else if (!blobs_file && magic == MAGIC_PSO) {
            uint32_t kind = in.u32(), sz = in.u32(); const uint8_t* p = in.raw(sz);
            if (!p) break;
            (seed ? r->seed_psos : r->psos).push_back({kind, {p, sz}});
        } else break;   // truncated tail from a crash: stop here
    }
}

#include "precompile.inc"

static DWORD WINAPI replay_main(void*) {
    Replay* r = g_rp;
    ULONGLONG t0 = GetTickCount64();
    ov_prepare();
    load_records(r, L"blobs.bin", true);
    load_records(r, L"psos.bin", false);
    load_records(r, L"blobs.bin", true, true);
    load_records(r, L"psos.bin", false, true);
    // the replay rebuilds this PC's recordings plus the seed's pipelines not recorded here yet
    r->todo = r->psos; r->todo_own = r->psos.size();
    {
        std::unordered_set<uint64_t> have;
        for (auto& p : r->psos) have.insert(fnv(p.second.first, p.second.second) ^ p.first);
        size_t extra = 0;
        for (auto& p : r->seed_psos) {
            if (have.insert(fnv(p.second.first, p.second.second) ^ p.first).second) { r->todo.push_back(p); extra++; }
            if (p.first != PSO_GFX && p.first != PSO_CS) continue;
            Reader in{p.second.first, p.second.second}; in.u64();   // root signature (shipped in the seed)
            for (int k = 0; k < (p.first == PSO_GFX ? 5 : 1); k++) { uint64_t h = in.u64(); if (h && !r->blobs.count(h)) r->wanted.insert(h); }
        }
        if (!r->seed_psos.empty())
            logf("seed: %zu pipelines from WH_SHADERS\\seed (%zu not in this PC's recordings yet, %zu of their shaders still to find in the game files); "
                 "used as render states for the pre-compile and rebuilt by the replay", r->seed_psos.size(), extra, r->wanted.size());
    }
    if (g_cfg_replay && r->todo.empty()) logf("nothing recorded yet");
    else if (g_cfg_replay) {
        run_replay(r, "replay");
        logf("replay done in %.1fs: ok=%ld failed=%ld skipped(missing data)=%ld; %ld of %ld pipelines recorded here earlier had to be compiled again",
             (GetTickCount64() - t0) / 1000.0, r->ok.load(), r->failed.load(), r->missing.load(), r->own_slow.load(), r->own_ok.load());
    }
    if (g_cfg_precompile) precompile(r);
    else {   // replay only: say so when it is done, and log what the game still compiles from here on
        ov_set(OV_DONE, 0, 0);
        log_pso_times("before the replay finished");
        g_pre_finished = 1;
    }
    // all done: let go of the game's device, so the game can create a new one after a driver reset,
    // and of the recordings read into memory
    for (auto& kv : r->roots) if (kv.second) kv.second->Release();
    r->roots.clear();
    r->dev->Release(); r->dev = nullptr;
    r->blobs.clear(); r->psos.clear(); r->seed_psos.clear(); r->todo.clear(); r->wanted.clear();
    std::vector<std::vector<uint8_t>>().swap(r->files);
    return 0;
}

static void start_replay(ID3D12Device* dev) {
    if (!g_cfg_replay && !g_cfg_precompile) return;
    g_rp = new Replay();   // kept for the life of the process
    dev->AddRef(); g_rp->dev = dev;
    CloseHandle(CreateThread(nullptr, 0, replay_main, nullptr, 0, nullptr));
}

// ------------------------------------------------------------------ setup
static bool writable(const std::wstring& dir) {
    CreateDirectoryW(dir.c_str(), nullptr);
    HANDLE h = CreateFileW((dir + L"write_test.tmp").c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_FLAG_DELETE_ON_CLOSE, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;
    CloseHandle(h);
    return true;
}

static void open_files() {
    // settings and seed live in WH_SHADERS next to this DLL (the game's exe folder)
    HMODULE self = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCWSTR)&open_files, &self);
    wchar_t dll[MAX_PATH]; GetModuleFileNameW(self, dll, MAX_PATH);
    std::wstring dll_dir(dll); dll_dir = dll_dir.substr(0, dll_dir.find_last_of(L"\\/") + 1);
    std::wstring dir_lc(dll_dir);   // for ids: the same folder however the game was started
    for (wchar_t& c : dir_lc) c = towlower(c);
    std::wstring root = dll_dir + L"WH_SHADERS\\";
    g_seed_dir = root + L"seed\\";
    std::wstring ini = root + L"WH_PSO.ini";
    // cache and logs too, unless the game folder is read-only (e.g. under Program Files)
    std::wstring data = root;
    bool moved = false;
    if (!writable(root)) {
        wchar_t la[MAX_PATH] = L"";
        DWORD ln = GetEnvironmentVariableW(L"LOCALAPPDATA", la, MAX_PATH);
        if (ln > 0 && ln < MAX_PATH) {
            wchar_t id[20]; swprintf(id, 20, L"%016llx", (unsigned long long)fnv(dir_lc.data(), dir_lc.size() * 2));
            std::wstring base = std::wstring(la) + L"\\WH_SHADERS\\";
            CreateDirectoryW(base.c_str(), nullptr);
            data = base + id + L"\\";
            moved = writable(data);
        }
        if (!moved) data = root;
    }
    g_dir = data + L"cache\\";
    CreateDirectoryW(g_dir.c_str(), nullptr);
    if (GetFileAttributesW(ini.c_str()) == INVALID_FILE_ATTRIBUTES)
        if (FILE* f = _wfopen(ini.c_str(), L"wb")) { fputs(INI_DEFAULT, f); fclose(f); }
    g_ov_enabled = GetPrivateProfileIntW(L"Overlay", L"Enabled", 1, ini.c_str()) != 0;
    g_cfg_replay = GetPrivateProfileIntW(L"Replay", L"Enabled", 1, ini.c_str()) != 0;
    g_cfg_precompile = GetPrivateProfileIntW(L"Precompile", L"Enabled", 1, ini.c_str()) != 0;
    g_live_taken = !g_cfg_precompile;   // only the pre-compile learns from them
    bool log_on = GetPrivateProfileIntW(L"Logging", L"Enabled", 0, ini.c_str()) != 0;
    g_log_level = log_on ? (int)GetPrivateProfileIntW(L"Logging", L"Level", 3, ini.c_str()) : 0;
    g_log_progress = (int)GetPrivateProfileIntW(L"Logging", L"ProgressEvery", 500, ini.c_str());
    g_log_max = (long long)GetPrivateProfileIntW(L"Logging", L"MaxSizeMB", 200, ini.c_str()) << 20;
    // 0 Idle, 1 Lowest, 2 BelowNormal, 3 Normal, 4 AboveNormal, 5 Highest
    static const int PRIO[] = {THREAD_PRIORITY_IDLE, THREAD_PRIORITY_LOWEST, THREAD_PRIORITY_BELOW_NORMAL,
                               THREAD_PRIORITY_NORMAL, THREAD_PRIORITY_ABOVE_NORMAL, THREAD_PRIORITY_HIGHEST};
    auto prio = [&](const wchar_t* key, int* level) {
        wchar_t t[16] = L"";
        GetPrivateProfileStringW(L"Performance", key, L"3", t, 16, ini.c_str());
        int v = (t[0] >= L'0' && t[0] <= L'5' && t[1] == 0) ? t[0] - L'0' : 3;   // anything else is 3 (Normal)
        *level = v;
        return PRIO[v];
    };
    int replay_lvl, pre_lvl;
    g_cfg_replay_prio = prio(L"ReplayPriority", &replay_lvl);
    g_cfg_pre_prio = prio(L"PrecompilePriority", &pre_lvl);
    g_cfg_replay_threads = (int)GetPrivateProfileIntW(L"Performance", L"ReplayThreads", -2, ini.c_str());
    g_cfg_pre_threads = (int)GetPrivateProfileIntW(L"Performance", L"PrecompileThreads", -2, ini.c_str());

    wchar_t exe[MAX_PATH]; GetModuleFileNameW(nullptr, exe, MAX_PATH);
    if (log_on) {   // a fresh wh_pso.log for every new DLL build, appended to while the DLL stays the same
        std::wstring logs = data + L"logs\\";
        CreateDirectoryW(logs.c_str(), nullptr);
        std::vector<uint8_t> me = read_file(dll);
        char build[64];
        snprintf(build, sizeof build, "%s %016llx", WH_PSO_VERSION, (unsigned long long)fnv(me.data(), me.size()));
        std::vector<uint8_t> prev = read_file(logs + L"dll_version.txt");
        std::string prev_s(prev.begin(), prev.end());
        while (!prev_s.empty() && (prev_s.back() == '\n' || prev_s.back() == '\r')) prev_s.pop_back();
        g_log_path = logs + L"wh_pso.log";
        if (prev_s != build) {
            DeleteFileW(g_log_path.c_str());
            DeleteFileW((logs + L"wh_pso.old.log").c_str());
            if (FILE* f = _wfopen((logs + L"dll_version.txt").c_str(), L"w")) { fprintf(f, "%s\n", build); fclose(f); }
        }
        g_log = _wfopen(g_log_path.c_str(), L"a");
        if (prev_s != build) logf("==== new log for dxgi.dll v%s (build %s)", WH_PSO_VERSION, build + strlen(WH_PSO_VERSION) + 1);
    }
    // remember what is already recorded so we only append new entries
    std::vector<uint8_t> f = read_file(g_dir + L"blobs.bin");
    size_t good = 0;
    for (Reader in{f.data(), f.size()}; in.o + 12 <= in.n;) {
        if (in.u32() != MAGIC_BLOB) break;
        in.u32(); uint64_t h = in.u64(); uint32_t sz = in.u32();
        if (!in.raw(sz)) break;
        g_blob_seen.insert(h); good = in.o;
    }
    // a second copy of the game (e.g. the trial exe at the same time) only forwards calls
    std::wstring dir_id(g_dir);
    for (wchar_t& c : dir_id) c = towlower(c);
    wchar_t mx[64]; swprintf(mx, 64, L"Local\\WH_PSO_%016llx", (unsigned long long)fnv(dir_id.data(), dir_id.size() * 2));
    CreateMutexW(nullptr, FALSE, mx);
    if (GetLastError() == ERROR_ALREADY_EXISTS) {
        g_cfg_replay = g_cfg_precompile = false; g_ov_enabled = false; g_live_taken = true;
        logf("---- v%s started in %ls: another copy of the game is already using this folder; only passing calls through", WH_PSO_VERSION, exe);
        AcquireSRWLockExclusive(&g_log_lock);   // the log belongs to the first copy
        if (g_log) { fclose(g_log); g_log = nullptr; }
        ReleaseSRWLockExclusive(&g_log_lock);
        return;
    }
    auto open_rec = [](const std::wstring& path, size_t good) {   // append after the last whole record, cutting off a torn tail
        FILE* f = _wfopen(path.c_str(), L"r+b");
        if (!f) f = _wfopen(path.c_str(), L"wb");
        if (f) { _fseeki64(f, (long long)good, SEEK_SET); SetEndOfFile((HANDLE)_get_osfhandle(_fileno(f))); }
        return f;
    };
    g_blobs = open_rec(g_dir + L"blobs.bin", good);
    f = read_file(g_dir + L"psos.bin"); good = 0; size_t n = 0;
    for (Reader in{f.data(), f.size()}; in.o + 12 <= in.n;) {
        if (in.u32() != MAGIC_PSO) break;
        uint32_t kind = in.u32(), sz = in.u32(); const uint8_t* p = in.raw(sz);
        if (!p) break;
        g_pso_seen.insert(fnv(p, sz) ^ kind); good = in.o; n++;
    }
    g_psos = open_rec(g_dir + L"psos.bin", good);
    logf("---- v%s started in %ls; %zu pipelines and %zu blobs already recorded", WH_PSO_VERSION, exe, n, g_blob_seen.size());
    if (moved) logf("the game folder is read-only; cache and logs are in %ls", data.c_str());
    else if (!g_blobs || !g_psos) logf("could not write to %ls: nothing is recorded or remembered between launches", g_dir.c_str());
    OSVERSIONINFOW os = {}; os.dwOSVersionInfoSize = sizeof os;
    typedef LONG(WINAPI * PFN_RGV)(OSVERSIONINFOW*);
    if (auto rgv = (PFN_RGV)GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "RtlGetVersion")) rgv(&os);
    logf("Windows %lu.%lu build %lu, %lu CPU threads in %u group(s)", os.dwMajorVersion, os.dwMinorVersion, os.dwBuildNumber,
         GetActiveProcessorCount(ALL_PROCESSOR_GROUPS), (unsigned)GetActiveProcessorGroupCount());
    logf("settings: overlay %s, replay %s (priority %d, threads %d), precompile %s (priority %d, threads %d), log level %d; threads -2 = idle CPU threads",
         g_ov_enabled ? "on" : "off", g_cfg_replay ? "on" : "off", replay_lvl, g_cfg_replay_threads,
         g_cfg_precompile ? "on" : "off", pre_lvl, g_cfg_pre_threads, g_log_level);
}

static void on_factory() {
    if (g_hook_state.load() == 0) install_hooks();
}

static HRESULT after_factory(HRESULT hr, void** f) {
    if (SUCCEEDED(hr) && f && *f && g_hook_state.load() == 2) ov_hook_factory(*f);
    return hr;
}
extern "C" HRESULT WINAPI My_CreateDXGIFactory(REFIID r, void** f) {
    if (!real_CreateDXGIFactory) return DXGI_ERROR_UNSUPPORTED;
    on_factory(); return after_factory(real_CreateDXGIFactory(r, f), f);
}
extern "C" HRESULT WINAPI My_CreateDXGIFactory1(REFIID r, void** f) {
    if (!real_CreateDXGIFactory1) return DXGI_ERROR_UNSUPPORTED;
    on_factory(); return after_factory(real_CreateDXGIFactory1(r, f), f);
}
extern "C" HRESULT WINAPI My_CreateDXGIFactory2(UINT fl, REFIID r, void** f) {
    if (!real_CreateDXGIFactory2) return DXGI_ERROR_UNSUPPORTED;
    on_factory(); return after_factory(real_CreateDXGIFactory2(fl, r, f), f);
}

BOOL WINAPI DllMain(HINSTANCE h, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(h);
        load_real();
    } else if (reason == DLL_PROCESS_DETACH) {
        g_exiting = true;
        inflight_close();   // a normal exit: the pre-compile didn't crash the game
        log_pso_times(g_pre_finished ? "after the pre-compile finished" : "this launch");
        logf("exit: recorded %ld new pipelines (%ld repeats, %ld unrecordable)", g_recorded.load(), g_dupes.load(), g_skipped.load());
    }
    return TRUE;
}
