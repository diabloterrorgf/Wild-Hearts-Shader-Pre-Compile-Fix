// ------------------------------------------------------------------ startup pre-compile
// Included from whpso.cpp. Finds every shader the game ships (the archives' shader
// containers, plus anything the one-off deep scan finds elsewhere, plus the exe),
// pairs vertex and pixel shaders the way each container groups them, and builds a
// pipeline for each pair from inside the game process, using the root signature and
// render state of a pipeline the game itself built (a "template"). Pipelines the game
// already built exactly are replayed instead (see replay_main). Progress is kept in
// cache\synth_done_<exe>.bin for the current GPU and driver, so later launches only
// build what is new.
#include <zlib.h>
#include <array>

static const uint32_t G2S_TYPE = 0x7BCD279F;   // the archives' shader containers ("_S2G")
static const uint64_t MAX_ENTRY = 256u << 20;   // largest entry worth unpacking; shader containers are a few MB

// ---- shader bytecode reflection (DXBC containers, shader model 5 and 6)
struct SigEl { std::string name; uint32_t idx, sv, ctype, reg; uint8_t mask; };
// stage: 0 ps 1 vs 2 gs 3 hs 4 ds 5 cs; pc: patch constants (HS out / DS in); dxil: shader model 6
struct ShaderInfo { int stage = -1; bool dxil = false; std::vector<SigEl> in, out, pc; };

static bool dxbc_chunk(const uint8_t* b, size_t n, uint32_t tag, const uint8_t** p, uint32_t* sz) {
    if (n < 32) return false;
    uint32_t cnt; memcpy(&cnt, b + 28, 4);
    for (uint32_t k = 0; k < cnt && 32 + 4 * k + 4 <= n; k++) {
        uint32_t o; memcpy(&o, b + 32 + 4 * k, 4);
        if ((size_t)o + 8 > n) return false;
        uint32_t t, s; memcpy(&t, b + o, 4); memcpy(&s, b + o + 4, 4);
        if (t == tag && (size_t)o + 8 + s <= n) { *p = b + o + 8; *sz = s; return true; }
    }
    return false;
}
#define FOURCC4(a, b, c, d) ((uint32_t)(a) | ((uint32_t)(b) << 8) | ((uint32_t)(c) << 16) | ((uint32_t)(d) << 24))

static bool parse_sig(const uint8_t* d, uint32_t n, int stride, int base, std::vector<SigEl>& out) {
    if (n < 8) return false;
    uint32_t cnt; memcpy(&cnt, d, 4);
    for (uint32_t i = 0; i < cnt; i++) {
        size_t p = 8 + (size_t)i * stride + base;
        if (p + 24 > n) return false;
        uint32_t v[5]; memcpy(v, d + p, 20);
        SigEl e;
        if (v[0] >= n) return false;
        const char* nm = (const char*)d + v[0];
        e.name.assign(nm, strnlen(nm, n - v[0]));
        e.idx = v[1]; e.sv = v[2]; e.ctype = v[3]; e.reg = v[4]; e.mask = d[p + 20];
        out.push_back(e);
    }
    return true;
}

// the stage is in the top 16 bits of the program's version, the same numbering for SM5 and SM6
static int shader_stage(const uint8_t* b, size_t n, bool* dxil) {
    const uint8_t* p; uint32_t sz; uint32_t v;
    *dxil = false;
    if (dxbc_chunk(b, n, FOURCC4('S', 'H', 'E', 'X'), &p, &sz) || dxbc_chunk(b, n, FOURCC4('S', 'H', 'D', 'R'), &p, &sz)) {
        if (sz < 4) return -1;
    } else if (dxbc_chunk(b, n, FOURCC4('D', 'X', 'I', 'L'), &p, &sz)) {
        if (sz < 4) return -1;
        *dxil = true;
    } else return -1;
    memcpy(&v, p, 4);
    int st = (int)(v >> 16);
    return st <= 5 ? st : -1;
}

static ShaderInfo shader_info(const uint8_t* b, size_t n) {
    ShaderInfo si; const uint8_t* p; uint32_t sz;
    si.stage = shader_stage(b, n, &si.dxil);
    if (dxbc_chunk(b, n, FOURCC4('I', 'S', 'G', 'N'), &p, &sz)) parse_sig(p, sz, 24, 0, si.in);
    else if (dxbc_chunk(b, n, FOURCC4('I', 'S', 'G', '1'), &p, &sz)) parse_sig(p, sz, 32, 4, si.in);
    if (dxbc_chunk(b, n, FOURCC4('O', 'S', 'G', 'N'), &p, &sz)) parse_sig(p, sz, 24, 0, si.out);
    else if (dxbc_chunk(b, n, FOURCC4('O', 'S', 'G', '5'), &p, &sz)) parse_sig(p, sz, 28, 4, si.out);
    else if (dxbc_chunk(b, n, FOURCC4('O', 'S', 'G', '1'), &p, &sz)) parse_sig(p, sz, 32, 4, si.out);
    if (dxbc_chunk(b, n, FOURCC4('P', 'C', 'S', 'G'), &p, &sz)) parse_sig(p, sz, 24, 0, si.pc);
    else if (dxbc_chunk(b, n, FOURCC4('P', 'S', 'G', '1'), &p, &sz)) parse_sig(p, sz, 32, 4, si.pc);
    return si;
}

// hull -> domain: control point and patch constant signatures must match exactly
static bool same_sig(const std::vector<SigEl>& a, const std::vector<SigEl>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++)
        if (a[i].idx != b[i].idx || a[i].reg != b[i].reg || a[i].mask != b[i].mask || a[i].ctype != b[i].ctype ||
            _stricmp(a[i].name.c_str(), b[i].name.c_str())) return false;
    return true;
}

// can this VS feed this PS? (same semantic in the same register, and every component the PS reads is written)
static bool linkable(const std::vector<SigEl>& vs_out, const std::vector<SigEl>& ps_in) {
    for (const SigEl& e : ps_in) {
        if (e.sv == 7 || e.sv == 9 || e.sv == 10) continue;   // PRIMITIVE_ID, IS_FRONT_FACE, SAMPLE_INDEX
        const SigEl* o = nullptr;
        for (const SigEl& v : vs_out)
            if (v.idx == e.idx && _stricmp(v.name.c_str(), e.name.c_str()) == 0) { o = &v; break; }
        // the driver needs every component the input declares to be written, not just the ones it reads
        if (!o || o->reg != e.reg || (e.mask & ~o->mask)) return false;
        if (o->ctype && e.ctype && o->ctype != e.ctype) return false;   // e.g. uint TEXCOORD into a float input: the driver rejects it
    }
    return true;
}

static void target_sig(const ShaderInfo& ps, int* n, uint32_t* types) {
    *n = 0; *types = 0;
    for (const SigEl& e : ps.out)
        if (e.sv == 64 && e.reg < 16) { if ((int)e.reg + 1 > *n) *n = (int)e.reg + 1; *types |= (e.ctype & 3) << (2 * e.reg); }
}

static uint64_t sem_key(const std::string& name, uint32_t idx) {
    std::string u(name); for (char& c : u) c = (char)toupper((unsigned char)c);
    return fnv(u.data(), u.size()) ^ ((uint64_t)idx << 56);
}

static uint64_t shader_key(const void* p, size_t n) { return blob_key(BLOB_SHADER, p, n); }

static void split_dxbc(const uint8_t* d, size_t n, std::vector<std::pair<const uint8_t*, uint32_t>>& out) {
    size_t i = 0;
    while (i + 32 <= n) {
        const uint8_t* f = (const uint8_t*)memchr(d + i, 'D', n - i);
        if (!f) break;
        i = f - d;
        if (i + 32 <= n && memcmp(f, "DXBC", 4) == 0) {
            uint32_t one, total, cnt; memcpy(&one, f + 20, 4); memcpy(&total, f + 24, 4); memcpy(&cnt, f + 28, 4);
            if (one == 1 && total > 32 && total <= n - i && cnt > 0 && cnt < 64) { out.push_back({f, total}); i += total; continue; }
        }
        i++;
    }
}

// ---- archives
struct G2SEntry { uint32_t arc; uint64_t off, esz, dsz; };
struct ArcInfo { std::wstring name; uint64_t size, mtime; };

static bool inflate_scan(const uint8_t* b, size_t n, uint8_t* out, size_t dsz) {
    // fallback: find each zlib stream by its header and inflate it
    size_t got = 0, i = 0;
    while (got < dsz && i + 2 < n) {
        if (!(b[i] == 0x78 && (b[i + 1] == 0x9c || b[i + 1] == 0xda || b[i + 1] == 0x01 || b[i + 1] == 0x5e))) { i++; continue; }
        z_stream z = {}; inflateInit(&z);
        z.next_in = (Bytef*)b + i; z.avail_in = (uInt)(n - i);
        z.next_out = out + got; z.avail_out = (uInt)(dsz - got);
        int rc = inflate(&z, Z_FINISH);
        size_t produced = (dsz - got) - z.avail_out, used = (n - i) - z.avail_in;
        inflateEnd(&z);
        if (rc == Z_STREAM_END) { got += produced; i += used; } else i++;
    }
    return got == dsz;
}

// entry body after the 0x30 header: 8 bytes, then chunks of [u16 compressed size][8 bytes][zlib data]
static bool inflate_entry(const uint8_t* b, size_t n, uint8_t* out, size_t dsz) {
    size_t p = 8, got = 0;
    while (got < dsz) {
        if (p + 10 > n) return inflate_scan(b, n, out, dsz);
        uint16_t clen; memcpy(&clen, b + p, 2); p += 10;
        if (p + clen > n) return inflate_scan(b, n, out, dsz);
        if (b[p] == 0x78) {
            uLongf o = (uLongf)(dsz - got);
            if (uncompress(out + got, &o, b + p, clen) != Z_OK) return inflate_scan(b, n, out, dsz);
            got += o;
        } else {
            size_t k = std::min<size_t>(clen, dsz - got);
            memcpy(out + got, b + p, k); got += k;
        }
        p += clen;
    }
    return got == dsz;
}

// whole entry (header included) -> its decompressed data
static bool unpack_entry(const std::vector<uint8_t>& raw, uint64_t esz, uint64_t dsz, std::vector<uint8_t>& data) {
    data.resize((size_t)dsz);
    uint64_t csz; memcpy(&csz, raw.data() + 16, 8);
    if (csz == dsz && csz <= esz) { memcpy(data.data(), raw.data() + (esz - csz), (size_t)csz); return true; }
    return inflate_entry(raw.data() + 0x30, (size_t)esz - 0x30, data.data(), (size_t)dsz);
}

static std::wstring package_dir() {
    wchar_t exe[MAX_PATH]; GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::wstring d(exe); d = d.substr(0, d.find_last_of(L"\\/") + 1);
    return d + L"asset\\fdata_ms12\\package\\";
}

// ---- deep scan: every entry of every archive (and every other file next to the exe) is
// unpacked and searched for shader bytecode, so no shader the game ships is left out.
// It runs once, when the index is (re)built, on the pre-compile threads.
struct DeepGroup { std::string name; std::vector<std::vector<uint8_t>> blobs; };
struct DeepFile {
    std::wstring path, rel; uint64_t size = 0, mtime = 0; bool idrk = false;
    std::vector<G2SEntry> ents;          // arc filled in after the scan
    std::vector<DeepGroup> groups;       // shaders found outside IDRK entries
};
struct DeepStats {
    std::atomic<long long> bytes{0};
    std::atomic<long> files_done{0}, entries{0}, unreadable{0};
    SRWLOCK lock = SRWLOCK_INIT;
    std::unordered_map<uint32_t, std::array<long long, 3>> by_type;   // type -> entries, with shaders, bytes
};

// finds a shader container header in a stream fed in pieces: "DXBC", a 16-byte checksum, then
// version 1, a plausible size and chunk count (as split_dxbc checks), so random bytes don't count
struct DxbcFinder {
    uint32_t last = 0; int got = -1; uint8_t h[28]; bool hit = false;
    void feed(const uint8_t* p, size_t n) {
        for (size_t i = 0; i < n && !hit; i++) {
            if (got >= 0) {
                h[got++] = p[i];
                if (got < 28) continue;
                uint32_t one, total, cnt; memcpy(&one, h + 16, 4); memcpy(&total, h + 20, 4); memcpy(&cnt, h + 24, 4);
                hit = one == 1 && total > 32 && cnt > 0 && cnt < 64;
                got = -1; last = 0;
                continue;
            }
            last = (last << 8) | p[i];
            if (last == 0x44584243u) got = 0;   // "DXBC"
        }
    }
};

// sequential reader over [pos, end) of a file with a 4 MB window
struct BufRd {
    HANDLE f; uint64_t pos, end; std::vector<uint8_t>& b; size_t lo = 0, hi = 0;
    BufRd(HANDLE f_, uint64_t p, uint64_t e, std::vector<uint8_t>& buf) : f(f_), pos(p), end(e), b(buf) { if (b.size() < (4u << 20)) b.resize(4u << 20); }
    const uint8_t* need(size_t n) {
        if (hi - lo < n) {
            memmove(b.data(), b.data() + lo, hi - lo); hi -= lo; lo = 0;
            size_t want = (size_t)std::min<uint64_t>(b.size() - hi, end - pos);
            if (want) {
                LARGE_INTEGER li; li.QuadPart = (LONGLONG)pos; DWORD rd = 0;
                SetFilePointerEx(f, li, nullptr, FILE_BEGIN);
                if (!ReadFile(f, b.data() + hi, (DWORD)want, &rd, nullptr)) rd = 0;
                hi += rd; pos += rd;
            }
            if (hi - lo < n) return nullptr;
        }
        return b.data() + lo;
    }
    void skip(size_t n) { lo += n; }
};

// does this IDRK entry's data contain shader bytecode? Streams it, so huge entries are fine.
static bool entry_has_dxbc(HANDLE f, uint64_t off, uint64_t esz, uint64_t csz, uint64_t dsz, std::vector<uint8_t>& buf,
                           std::vector<uint8_t>& out, z_stream& zs, DeepStats& st, bool* readable) {
    DxbcFinder fd; *readable = true;
    if (csz == dsz && csz <= esz) {   // stored
        BufRd r(f, off + esz - csz, off + esz, buf);
        uint64_t left = csz;
        while (left && !fd.hit) {
            size_t n = (size_t)std::min<uint64_t>(left, 1u << 20);
            const uint8_t* p = r.need(n);
            if (!p) { *readable = false; break; }
            fd.feed(p, n); r.skip(n); left -= n;
        }
        st.bytes += (long long)csz;
        return fd.hit;
    }
    BufRd r(f, off + 0x30, off + esz, buf);
    if (out.size() < (256u << 10)) out.resize(256u << 10);
    uint64_t got = 0; bool ok = r.need(8) != nullptr;
    if (ok) r.skip(8);
    while (ok && got < dsz && !fd.hit) {
        const uint8_t* h = r.need(10);
        if (!h) { ok = false; break; }
        uint16_t clen; memcpy(&clen, h, 2); r.skip(10);
        const uint8_t* c = clen ? r.need(clen) : nullptr;
        if (!c) { ok = false; break; }
        if (c[0] == 0x78) {
            inflateReset(&zs);
            zs.next_in = (Bytef*)c; zs.avail_in = clen;
            int rc;
            do {
                zs.next_out = out.data(); zs.avail_out = (uInt)out.size();
                rc = inflate(&zs, Z_NO_FLUSH);
                size_t made = out.size() - zs.avail_out;
                fd.feed(out.data(), made); got += made;
            } while (rc == Z_OK && !fd.hit);
            if (rc != Z_STREAM_END && !fd.hit) ok = false;
        } else {
            size_t k = (size_t)std::min<uint64_t>(clen, dsz - got);
            fd.feed(c, k); got += k;
        }
        r.skip(clen);
    }
    st.bytes += (long long)got;
    if (fd.hit || ok) return fd.hit;
    // the chunk layout didn't parse: unpack the whole entry the slow way (as the pre-compile does)
    if (esz > MAX_ENTRY || dsz > MAX_ENTRY) { *readable = false; return false; }
    std::vector<uint8_t> raw((size_t)esz), data;
    LARGE_INTEGER li; li.QuadPart = (LONGLONG)off; DWORD rd = 0;
    SetFilePointerEx(f, li, nullptr, FILE_BEGIN);
    if (!ReadFile(f, raw.data(), (DWORD)esz, &rd, nullptr) || rd != esz || !unpack_entry(raw, esz, dsz, data)) { *readable = false; return false; }
    std::vector<std::pair<const uint8_t*, uint32_t>> parts;
    split_dxbc(data.data(), data.size(), parts);
    return !parts.empty();
}

static void deep_idrk(DeepFile& df, std::vector<uint8_t>& buf, std::vector<uint8_t>& out, z_stream& zs, DeepStats& st) {
    HANDLE f = CreateFileW(df.path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (f == INVALID_HANDLE_VALUE) { logd("deep scan: could not open %ls", df.rel.c_str()); st.unreadable++; return; }
    uint64_t off = df.rel.size() > 6 && df.rel.compare(df.rel.size() - 6, 6, L".fdata") == 0 ? 0x10 : 0;
    std::unordered_map<uint32_t, std::array<long long, 3>> local;
    while (off + 0x30 <= df.size) {
        uint8_t hd[0x30]; DWORD rd = 0; LARGE_INTEGER li; li.QuadPart = (LONGLONG)off;
        SetFilePointerEx(f, li, nullptr, FILE_BEGIN);
        if (!ReadFile(f, hd, 0x30, &rd, nullptr) || rd != 0x30 || memcmp(hd, "IDRK0000", 8)) break;
        uint64_t esz, csz, dsz; uint32_t type;
        memcpy(&esz, hd + 8, 8); memcpy(&csz, hd + 16, 8); memcpy(&dsz, hd + 24, 8); memcpy(&type, hd + 0x28, 4);
        if (esz < 0x30 || off + esz > df.size) break;
        st.entries++;
        auto& row = local[type]; row[0]++;
        bool has;
        if (type == G2S_TYPE) has = true;   // shader containers: known, no need to unpack here
        else {
            bool readable;
            has = entry_has_dxbc(f, off, esz, csz, dsz, buf, out, zs, st, &readable);
            row[2] += (long long)dsz;
            if (!readable) { st.unreadable++; logd("deep scan: %ls entry at 0x%llx (type %08x, %llu bytes) could not be unpacked", df.rel.c_str(), (unsigned long long)off, type, (unsigned long long)dsz); }
        }
        if (has && esz <= MAX_ENTRY && dsz <= MAX_ENTRY) { row[1]++; df.ents.push_back({0, off, esz, dsz}); }
        else if (has) logf("deep scan: %ls entry at 0x%llx (type %08x) looks like it holds shaders but is too big to use (%llu bytes)",
                           df.rel.c_str(), (unsigned long long)off, type, (unsigned long long)dsz);
        off = (off + esz + 15) & ~15ULL;
        g_ov_bytes = st.bytes.load();
    }
    CloseHandle(f);
    AcquireSRWLockExclusive(&st.lock);
    for (auto& kv : local) { auto& r = st.by_type[kv.first]; for (int i = 0; i < 3; i++) r[i] += kv.second[i]; }
    ReleaseSRWLockExclusive(&st.lock);
}

// any other file: shader bytecode stored as-is, or inside any zlib stream
static void deep_generic(DeepFile& df, std::vector<uint8_t>& out, z_stream& zs, DeepStats& st) {
    HANDLE f = CreateFileW(df.path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (f == INVALID_HANDLE_VALUE) { logd("deep scan: could not open %ls", df.rel.c_str()); st.unreadable++; return; }
    HANDLE m = df.size ? CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr) : nullptr;
    const uint8_t* v = m ? (const uint8_t*)MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0) : nullptr;
    if (v) {
        size_t n = (size_t)df.size;
        std::vector<std::pair<const uint8_t*, uint32_t>> parts;
        split_dxbc(v, n, parts);
        if (!parts.empty()) {
            DeepGroup g; g.name = "@raw";
            for (auto& pr : parts) g.blobs.emplace_back(pr.first, pr.first + pr.second);
            df.groups.push_back(std::move(g));
        }
        std::vector<uint8_t> acc;
        size_t i = 0; long streams = 0;
        while (i + 2 <= n) {
            const uint8_t* q = (const uint8_t*)memchr(v + i, 0x78, n - i);
            if (!q) break;
            i = q - v;
            if (i + 2 > n || ((0x78u << 8) | q[1]) % 31 || (q[1] & 0x20)) { i++; continue; }
            inflateReset(&zs);
            zs.next_in = (Bytef*)q; zs.avail_in = (uInt)std::min<size_t>(n - i, 0x7FFFFFFF);
            acc.clear(); DxbcFinder fd; int rc;
            do {
                zs.next_out = out.data(); zs.avail_out = (uInt)out.size();
                rc = inflate(&zs, Z_NO_FLUSH);
                size_t made = out.size() - zs.avail_out;
                fd.feed(out.data(), made);
                if (acc.size() < (64u << 20)) acc.insert(acc.end(), out.data(), out.data() + made);
            } while (rc == Z_OK);
            if (rc == Z_STREAM_END && zs.total_out > 0) {
                streams++; st.bytes += (long long)zs.total_out;
                if (fd.hit) {
                    std::vector<std::pair<const uint8_t*, uint32_t>> ps;
                    split_dxbc(acc.data(), acc.size(), ps);
                    if (!ps.empty()) {
                        DeepGroup g; char nm[64]; snprintf(nm, sizeof nm, "@0x%llx", (unsigned long long)i); g.name = nm;
                        for (auto& pr : ps) g.blobs.emplace_back(pr.first, pr.first + pr.second);
                        df.groups.push_back(std::move(g));
                    }
                }
                i += std::max<size_t>(1, zs.total_in);
            } else i++;
            if (acc.capacity() > (16u << 20)) std::vector<uint8_t>().swap(acc);
            g_ov_bytes = st.bytes.load();
        }
        logv("deep scan: %ls: %ld zlib streams unpacked, %zu groups of shaders", df.rel.c_str(), streams, df.groups.size());
        UnmapViewOfFile(v);
    } else if (df.size) { st.unreadable++; logd("deep scan: could not map %ls", df.rel.c_str()); }
    if (m) CloseHandle(m);
    CloseHandle(f);
}

struct DeepJob { std::vector<DeepFile>* files; std::vector<size_t> order; std::atomic<size_t> next{0}; DeepStats st; };
static DWORD WINAPI deep_worker(void* arg) {
    DeepJob& J = *(DeepJob*)arg;
    SetThreadPriority(GetCurrentThread(), g_cfg_pre_prio);
    std::vector<uint8_t> buf, out(256u << 10);
    z_stream zs = {}; inflateInit(&zs);
    for (;;) {
        size_t k = J.next++;
        if (k >= J.order.size()) break;
        DeepFile& df = (*J.files)[J.order[k]];
        try { if (df.idrk) deep_idrk(df, buf, out, zs, J.st); else deep_generic(df, out, zs, J.st); }
        catch (...) { J.st.unreadable++; logf("deep scan: out of memory reading %ls; skipped", df.rel.c_str()); }
        long done = ++J.st.files_done;
        ov_set(OV_SCAN, done, (long)J.order.size());
        if (g_log_progress > 0 && done % g_log_progress == 0)
            logf("  deep scan %ld/%zu files, %.1f GB unpacked, %ld entries", done, J.order.size(), J.st.bytes / 1e9, J.st.entries.load());
    }
    inflateEnd(&zs);
    return 0;
}

static void list_files(const std::wstring& dir, const std::wstring& rel, std::vector<DeepFile>& out, int depth = 0) {
    WIN32_FIND_DATAW fd; HANDLE h = FindFirstFileW((dir + L"*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    do {
        std::wstring n = fd.cFileName;
        if (n == L"." || n == L"..") continue;
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            if (rel.empty() && _wcsicmp(n.c_str(), L"WH_SHADERS") == 0) continue;   // our own files
            // links (junctions, symlinks) can point outside the game folder or back into it
            if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) && depth < 16)
                list_files(dir + n + L"\\", rel + n + L"\\", out, depth + 1);
            continue;
        }
        DeepFile f; f.path = dir + n; f.rel = rel + n;
        f.size = ((uint64_t)fd.nFileSizeHigh << 32) | fd.nFileSizeLow;
        f.mtime = ((uint64_t)fd.ftLastWriteTime.dwHighDateTime << 32) | fd.ftLastWriteTime.dwLowDateTime;
        out.push_back(std::move(f));
    } while (FindNextFileW(h, &fd));
    FindClose(h);
}

static std::wstring exe_dir() {
    wchar_t exe[MAX_PATH]; GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::wstring d(exe); return d.substr(0, d.find_last_of(L"\\/") + 1);
}

// write to a temporary file, then swap it in, so a crash never leaves a half-written file
static void save_file(const std::wstring& path, const Buf& b) {
    std::wstring tmp = path + L".tmp";
    FILE* f = _wfopen(tmp.c_str(), L"wb");
    if (!f) return;
    bool ok = fwrite(b.v.data(), 1, b.v.size(), f) == b.v.size();
    ok = fclose(f) == 0 && ok;
    if (!ok || !MoveFileExW(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) DeleteFileW(tmp.c_str());
}

// deep_shaders.bin: shaders found outside IDRK entries, grouped by where they were found
static void save_deep(const std::vector<DeepFile>& files) {
    Buf b; b.u32(0x50454544);   // "DEEP"
    uint32_t ng = 0; for (auto& f : files) ng += (uint32_t)f.groups.size();
    b.u32(ng);
    for (auto& f : files)
        for (auto& g : f.groups) {
            char nm[600]; snprintf(nm, sizeof nm, "%ls%s", f.rel.c_str(), g.name.c_str());
            b.u32((uint32_t)strlen(nm)); b.raw(nm, strlen(nm));
            b.u32((uint32_t)g.blobs.size());
            for (auto& x : g.blobs) { b.u32((uint32_t)x.size()); b.raw(x.data(), x.size()); }
        }
    save_file(g_dir + L"deep_shaders.bin", b);
}

static void scan_archives(std::vector<ArcInfo>& arcs, std::vector<G2SEntry>& ents) {
    std::wstring root = exe_dir(), pkg = package_dir();
    std::vector<DeepFile> files;
    list_files(root, L"", files);
    wchar_t self[MAX_PATH]; HMODULE me = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCWSTR)&scan_archives, &me);
    GetModuleFileNameW(me, self, MAX_PATH);
    wchar_t exe[MAX_PATH]; GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::vector<DeepFile> keep;
    for (DeepFile& f : files) {
        if (_wcsicmp(f.path.c_str(), self) == 0 || _wcsicmp(f.path.c_str(), exe) == 0) continue;   // this DLL; the exe is read separately
        // IDRK archives live under the package folder; their names are kept relative to it
        if (f.path.size() > pkg.size() && _wcsnicmp(f.path.c_str(), pkg.c_str(), pkg.size()) == 0) {
            HANDLE h = CreateFileW(f.path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
            char m[8] = {}; DWORD rd = 0;
            if (h != INVALID_HANDLE_VALUE) {
                if (f.rel.size() > 6 && f.rel.compare(f.rel.size() - 6, 6, L".fdata") == 0) { LARGE_INTEGER li; li.QuadPart = 0x10; SetFilePointerEx(h, li, nullptr, FILE_BEGIN); }
                ReadFile(h, m, 8, &rd, nullptr); CloseHandle(h);
            }
            if (rd == 8 && !memcmp(m, "IDRK0000", 8)) { f.idrk = true; f.rel = f.path.substr(pkg.size()); }
        }
        keep.push_back(std::move(f));
    }
    files.swap(keep);
    long long total = 0; int idrk = 0;
    for (auto& f : files) { total += (long long)f.size; idrk += f.idrk; }
    DeepJob* J = new DeepJob(); J->files = &files;
    for (size_t i = 0; i < files.size(); i++) J->order.push_back(i);
    // biggest first so one huge file doesn't finish last on a single thread
    std::sort(J->order.begin(), J->order.end(), [&](size_t a, size_t b) { return files[a].size > files[b].size; });
    ULONGLONG t0 = GetTickCount64();
    g_ov_bytes = 0; ov_set(OV_SCAN, 0, (long)files.size());
    std::vector<HANDLE> th;
    int n = start_threads(g_cfg_pre_threads, deep_worker, J, th, "deep scan");
    logf("deep scan: unpacking every entry of %d archives and searching %zu other files (%.1f GB on disk) for shaders, on %d threads",
         idrk, files.size() - idrk, total / 1e9, n);
    if (!th.empty()) WaitForMultipleObjects((DWORD)th.size(), th.data(), TRUE, INFINITE);
    for (HANDLE h : th) CloseHandle(h);
    // archives (sorted by name) and their entries holding shaders
    std::vector<size_t> by_name;
    for (size_t i = 0; i < files.size(); i++) if (files[i].idrk && !files[i].ents.empty()) by_name.push_back(i);
    std::sort(by_name.begin(), by_name.end(), [&](size_t a, size_t b) { return files[a].rel < files[b].rel; });
    for (size_t i : by_name) {
        uint32_t arc = (uint32_t)arcs.size();
        arcs.push_back({files[i].rel, files[i].size, files[i].mtime});
        for (G2SEntry e : files[i].ents) { e.arc = arc; ents.push_back(e); }
    }
    save_deep(files);
    DeepStats& st = J->st;
    long other_shaders = 0;
    for (auto& f : files) for (auto& g : f.groups) { other_shaders += (long)g.blobs.size(); logf("deep scan: %ld shaders in %ls%s", (long)g.blobs.size(), f.rel.c_str(), g.name.c_str()); }
    logf("deep scan done in %.0fs: %.1f GB unpacked from %ld archive entries; %zu entries hold shaders; %ld shaders found in other files; %ld entries could not be unpacked",
         (GetTickCount64() - t0) / 1000.0, st.bytes / 1e9, st.entries.load(), ents.size(), other_shaders, st.unreadable.load());
    std::vector<std::pair<uint32_t, std::array<long long, 3>>> rows(st.by_type.begin(), st.by_type.end());
    std::sort(rows.begin(), rows.end(), [](const std::pair<uint32_t, std::array<long long, 3>>& a, const std::pair<uint32_t, std::array<long long, 3>>& b) { return a.second[0] > b.second[0]; });
    for (auto& r : rows)
        logf("deep scan:   entry type %08x: %6lld entries, %7.2f GB unpacked, %5lld with shaders%s", r.first, r.second[0], r.second[2] / 1e9, r.second[1],
             r.first == G2S_TYPE ? " (shader containers, not unpacked here)" : "");
    delete J;
}

static bool arc_unchanged(const ArcInfo& a) {
    WIN32_FILE_ATTRIBUTE_DATA d;
    if (!GetFileAttributesExW((package_dir() + a.name).c_str(), GetFileExInfoStandard, &d)) return false;
    uint64_t sz = ((uint64_t)d.nFileSizeHigh << 32) | d.nFileSizeLow;
    uint64_t mt = ((uint64_t)d.ftLastWriteTime.dwHighDateTime << 32) | d.ftLastWriteTime.dwLowDateTime;
    return sz == a.size && mt == a.mtime;
}

static int count_archives() {
    WIN32_FIND_DATAW fd; int n = 0; HANDLE h = FindFirstFileW((package_dir() + L"*.fdata").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return 0;
    do n++; while (FindNextFileW(h, &fd));
    FindClose(h);
    return n;
}

// index.bin caches which archive entries hold shaders, so later launches skip the deep scan
static const uint32_t INDEX_MAGIC = 0x34444E49;   // "IND4": v1.12.0 deep scan (stricter shader check)
static bool load_index(std::vector<ArcInfo>& arcs, std::vector<G2SEntry>& ents) {
    if (GetFileAttributesW((g_dir + L"deep_shaders.bin").c_str()) == INVALID_FILE_ATTRIBUTES) return false;
    std::vector<uint8_t> f = read_file(g_dir + L"index.bin");
    Reader in{f.data(), f.size()};
    if (in.u32() != INDEX_MAGIC || (int)in.u32() != count_archives()) return false;
    uint32_t na = in.u32();
    for (uint32_t i = 0; i < na && !in.bad; i++) {
        uint32_t l = in.u32();
        if (l > 32767) return false;
        const uint8_t* s = in.raw((size_t)l * 2);
        if (!s) return false;
        ArcInfo a; a.name.assign((const wchar_t*)s, l); a.size = in.u64(); a.mtime = in.u64();
        if (!arc_unchanged(a)) return false;
        arcs.push_back(a);
    }
    uint32_t ne = in.u32();
    for (uint32_t i = 0; i < ne && !in.bad; i++) {
        G2SEntry e; e.arc = in.u32(); e.off = in.u64(); e.esz = in.u64(); e.dsz = in.u64();
        if (e.arc >= na || e.esz < 0x30 || e.esz > MAX_ENTRY || e.dsz > MAX_ENTRY) return false;   // damaged file: scan again
        ents.push_back(e);
    }
    return !in.bad;
}

static void save_index(const std::vector<ArcInfo>& arcs, const std::vector<G2SEntry>& ents) {
    Buf b; b.u32(INDEX_MAGIC); b.u32((uint32_t)count_archives()); b.u32((uint32_t)arcs.size());
    for (const ArcInfo& a : arcs) { b.u32((uint32_t)a.name.size()); b.raw(a.name.data(), a.name.size() * 2); b.u64(a.size); b.u64(a.mtime); }
    b.u32((uint32_t)ents.size());
    for (const G2SEntry& e : ents) { b.u32(e.arc); b.u64(e.off); b.u64(e.esz); b.u64(e.dsz); }
    save_file(g_dir + L"index.bin", b);
}

// ---- templates: pipelines the game built, decoded
struct Elem { std::string name; uint32_t idx, fmt, slot, off, cls, step; };
struct GfxRec {
    uint64_t root = 0, vs = 0, ps = 0, ds = 0, hs = 0, gs = 0;
    bool has_so = false;
    uint8_t blend[328], rast[44], depth[52];
    uint32_t sample_mask = 0, strip = 0, topo = 0, nrt = 0, rtv[8] = {}, dsv = 0, sd[2] = {1, 0}, node = 0, flags = 0;
    std::vector<Elem> layout;
};
static bool decode_gfx(const uint8_t* p, size_t n, GfxRec& g) {
    Reader in{p, n};
    g.root = in.u64(); g.vs = in.u64(); g.ps = in.u64(); g.ds = in.u64(); g.hs = in.u64(); g.gs = in.u64();
    uint32_t nso = in.u32();
    for (uint32_t i = 0; i < nso && !in.bad; i++) { in.u32(); uint32_t l = in.u32(); in.raw(l); in.u32(); in.u32(); in.u32(); in.u32(); }
    uint32_t ns = in.u32(); for (uint32_t i = 0; i < ns && !in.bad; i++) in.u32();
    in.u32();
    g.has_so = nso > 0;
    if (auto q = in.raw(328)) memcpy(g.blend, q, 328);
    g.sample_mask = in.u32();
    if (auto q = in.raw(44)) memcpy(g.rast, q, 44);
    if (auto q = in.raw(52)) memcpy(g.depth, q, 52);
    uint32_t ne = in.u32();
    for (uint32_t i = 0; i < ne && !in.bad; i++) {
        Elem e; uint32_t l = in.u32(); const uint8_t* s = in.raw(l);
        if (s) e.name.assign((const char*)s, l);
        e.idx = in.u32(); e.fmt = in.u32(); e.slot = in.u32(); e.off = in.u32(); e.cls = in.u32(); e.step = in.u32();
        g.layout.push_back(e);
    }
    g.strip = in.u32(); g.topo = in.u32(); g.nrt = in.u32();
    if (auto q = in.raw(32)) memcpy(g.rtv, q, 32);
    g.dsv = in.u32();
    if (auto q = in.raw(8)) memcpy(g.sd, q, 8);
    g.node = in.u32(); g.flags = in.u32();
    return !in.bad;
}

struct Tmpl {
    GfxRec g; uint64_t uses = 0; int max_target = -1; ShaderInfo vs_info; bool vs_known = false;
    std::string family;                       // shader family of the container its shaders come from (e.g. PB2DS)
    int tgt_n = -1; uint32_t tgt_types = 0;   // the game's PS outputs: render target count and float/int per target
    std::vector<uint64_t> layout_keys;        // sorted semantic name+index hashes of its vertex layout
    std::unordered_map<std::string, int> votes;
};
struct CsTmpl { uint64_t root; uint64_t uses; };

typedef std::vector<std::pair<const uint8_t*, uint32_t>> Parts;   // shaders in one container or file

struct Pre {
    Replay* r;
    std::vector<Tmpl> gfx;                          // distinct render states, most used first
    std::vector<CsTmpl> cs;                         // distinct compute root signatures, most used first
    std::vector<Tmpl> tess;                         // distinct tessellation (VS+HS+DS[+PS]) states, most used first
    std::unordered_map<uint64_t, int> tess_by_shader;   // VS/HS/DS/PS key -> tessellation template the game used it with
    std::unordered_map<uint64_t, int> by_shader;    // VS or PS key -> template the game used it with
    std::unordered_map<uint64_t, uint64_t> cs_root; // CS key -> root signature the game used it with
    std::unordered_set<uint64_t> recorded;          // exact pairs already in the driver cache (game built them, or replay did)
    std::unordered_map<uint64_t, std::vector<uint8_t>> roots;   // root signature blobs seen this launch
    std::unordered_set<uint64_t> done;              // pipelines built on an earlier launch (same GPU and driver)
    std::unordered_set<uint64_t> shader_seen;       // shaders already handled this launch
    std::unordered_set<uint64_t> discovered;        // every shader found (overlay total)
    std::unordered_set<uint64_t> counted;           // of those, handled so far (overlay progress)
    std::atomic<long> processed{0};                 // counted.size()
    std::vector<ArcInfo> arcs; std::vector<G2SEntry> ents;
    std::vector<std::pair<std::string, Parts>> groups;   // shaders outside the archive containers: the exe, other files
    std::atomic<size_t> next{0};
    static const int variants = 3;                  // render states built per shader pair
    bool learn = false;                             // first pass: only learn template families
    std::atomic<long> no_position{0};
    std::atomic<long> built{0}, failed{0}, skipped{0}, bad_entries{0}, tess_built{0}, synth_built{0}, gs_built{0};
    std::unordered_set<uint64_t> gs_seen;           // geometry shaders handled this launch
    std::unordered_set<uint64_t> tess_seen;         // domain shaders handled this launch
    // Crash guard: each compile thread notes the pipeline it is building in cache\inflight_<exe>.bin. A
    // normal exit deletes that file; if it is still there at the next start, the game crashed or the
    // device was lost (maybe in the GPU driver, on a pipeline we asked for). A pipeline caught like
    // that twice, on the same GPU and driver, is skipped from then on.
    std::unordered_set<uint64_t> bad;
    std::atomic<int> slots{0};
    std::atomic<bool> lost_device{false};
    SRWLOCK lock = SRWLOCK_INIT;
    FILE* done_file = nullptr;
};
static HANDLE g_inflight = INVALID_HANDLE_VALUE;
static wchar_t g_inflight_path[2 * MAX_PATH];
static thread_local int t_slot = -1;
static void inflight_note(uint64_t k) {   // 0 = none
    if (t_slot < 0 || g_inflight == INVALID_HANDLE_VALUE) return;
    OVERLAPPED o = {}; o.Offset = 16 + (DWORD)t_slot * 8; DWORD w;
    WriteFile(g_inflight, &k, 8, &w, &o);
}
static void inflight_close() {   // the pre-compile ended or the game is closing normally
    if (g_inflight == INVALID_HANDLE_VALUE) return;
    CloseHandle(g_inflight); g_inflight = INVALID_HANDLE_VALUE;
    if (!g_dev_lost) DeleteFileW(g_inflight_path);   // kept when the device was lost: the next start counts it
}

static uint64_t pair_key(uint64_t vs, uint64_t ps) { return vs * 0x9E3779B97F4A7C15ULL ^ ps; }
static uint64_t tess_key(uint64_t vs, uint64_t hs, uint64_t ds, uint64_t ps) { return pair_key(pair_key(vs, hs), pair_key(ds, ps)) ^ 0x7E55; }

static std::string state_key(const GfxRec& g) {
    Buf b; b.u64(g.root); b.raw(g.blend, 328); b.u32(g.sample_mask); b.raw(g.rast, 44); b.raw(g.depth, 52);
    b.u32(g.strip); b.u32(g.topo); b.u32(g.nrt); b.raw(g.rtv, 32); b.u32(g.dsv); b.raw(g.sd, 8); b.u32(g.flags);
    for (const Elem& e : g.layout) { b.raw(e.name.data(), e.name.size()); b.u32(e.idx); b.u32(e.fmt); b.u32(e.slot); b.u32(e.off); b.u32(e.cls); }
    return std::string(b.v.begin(), b.v.end());
}

// templates from this PC's recordings, this session's pipelines and the seed. A pipeline is
// already in the driver cache (so the pre-compile skips its shader pair) when the game built it
// this session, or when the replay rebuilt it.
static void build_templates(Pre& P) {
    std::vector<std::vector<uint8_t>> gfx_recs, cs_recs;
    std::vector<bool> gfx_cached, cs_cached;
    auto add = [&](uint32_t kind, const uint8_t* p, uint32_t n, bool cached) {
        if (kind == PSO_GFX) { gfx_recs.emplace_back(p, p + n); gfx_cached.push_back(cached); }
        else if (kind == PSO_CS) { cs_recs.emplace_back(p, p + n); cs_cached.push_back(cached); }
    };
    for (auto& rec : P.r->psos) add(rec.first, rec.second.first, rec.second.second, g_cfg_replay);
    AcquireSRWLockExclusive(&g_lock);
    for (auto& v : g_live_gfx) add(PSO_GFX, v.data(), (uint32_t)v.size(), true);
    for (auto& v : g_live_cs) add(PSO_CS, v.data(), (uint32_t)v.size(), true);
    for (auto& kv : g_live_roots) P.roots[kv.first] = kv.second;
    g_live_taken = true;   // learned: stop collecting
    std::vector<std::vector<uint8_t>>().swap(g_live_gfx);
    std::vector<std::vector<uint8_t>>().swap(g_live_cs);
    g_live_roots.clear();
    ReleaseSRWLockExclusive(&g_lock);
    for (auto& rec : P.r->seed_psos) add(rec.first, rec.second.first, rec.second.second, g_cfg_replay);
    size_t gi = 0;

    std::unordered_map<std::string, int> idx;
    for (auto& v : gfx_recs) {
        bool own = gfx_cached[gi++];
        GfxRec g;
        if (!decode_gfx(v.data(), v.size(), g) || g.root == ~0ULL || !g.vs) continue;
        if (own) P.recorded.insert(pair_key(g.vs, g.ps));
        if (g.hs && g.ds && !g.gs && !g.has_so) {   // tessellation: kept as its own kind of template
            if (own) P.recorded.insert(tess_key(g.vs, g.hs, g.ds, g.ps));
            std::string k = "T" + state_key(g);
            auto it = idx.find(k); int t;
            if (it == idx.end()) {
                t = (int)P.tess.size(); idx[k] = t;
                Tmpl tm; tm.g = g; tm.uses = 0;
                auto b = P.r->blobs.find(g.vs);
                if (b != P.r->blobs.end()) { tm.vs_info = shader_info(b->second.first, b->second.second); tm.vs_known = true; }
                P.tess.push_back(tm);
            } else t = it->second;
            P.tess[t].uses++;
            for (uint64_t s : {g.vs, g.hs, g.ds, g.ps}) if (s) P.tess_by_shader[s] = t;
            continue;
        }
        if (g.has_so || g.hs || g.ds || g.gs || g.topo == 4 /* PATCH */) continue;   // not reusable for plain VS+PS
        std::string k = state_key(g);
        auto it = idx.find(k); int t;
        if (it == idx.end()) {
            t = (int)P.gfx.size(); idx[k] = t;
            Tmpl tm; tm.g = g; tm.uses = 0;
            auto b = P.r->blobs.find(g.vs);
            if (b != P.r->blobs.end()) { tm.vs_info = shader_info(b->second.first, b->second.second); tm.vs_known = true; }
            P.gfx.push_back(tm);
        } else t = it->second;
        P.gfx[t].uses++;
        P.by_shader[g.vs] = t;
        if (g.ps) P.by_shader[g.ps] = t;
    }
    std::unordered_map<uint64_t, uint64_t> cs_uses;
    size_t ci = 0;
    for (auto& v : cs_recs) {
        bool own = cs_cached[ci++];
        Reader in{v.data(), v.size()}; uint64_t root = in.u64(), cs = in.u64();
        if (in.bad || root == ~0ULL) continue;
        if (own) P.recorded.insert(pair_key(cs, 0));
        P.cs_root[cs] = root; cs_uses[root]++;
    }
    for (auto& kv : cs_uses) P.cs.push_back({kv.first, kv.second});
    std::sort(P.cs.begin(), P.cs.end(), [](const CsTmpl& a, const CsTmpl& b) { return a.uses > b.uses; });
    // order render states by how often the game uses them (by_shader indices must follow)
    std::vector<int> order(P.gfx.size());
    for (size_t i = 0; i < order.size(); i++) order[i] = (int)i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return P.gfx[a].uses > P.gfx[b].uses; });
    std::vector<int> where(order.size());
    std::vector<Tmpl> sorted;
    for (size_t i = 0; i < order.size(); i++) { where[order[i]] = (int)i; sorted.push_back(P.gfx[order[i]]); }
    P.gfx.swap(sorted);
    for (auto& kv : P.by_shader) kv.second = where[kv.second];
    std::vector<int> tord(P.tess.size());
    for (size_t i = 0; i < tord.size(); i++) tord[i] = (int)i;
    std::stable_sort(tord.begin(), tord.end(), [&](int a, int b) { return P.tess[a].uses > P.tess[b].uses; });
    std::vector<int> twhere(tord.size()); std::vector<Tmpl> tsorted;
    for (size_t i = 0; i < tord.size(); i++) { twhere[tord[i]] = (int)i; tsorted.push_back(P.tess[tord[i]]); }
    P.tess.swap(tsorted);
    for (auto& kv : P.tess_by_shader) kv.second = twhere[kv.second];
    for (auto* list : {&P.gfx, &P.tess}) for (Tmpl& t : *list) {
        t.max_target = -1; for (int i = 0; i < 8; i++) if (i < (int)t.g.nrt && t.g.rtv[i]) t.max_target = i;
        for (const Elem& e : t.g.layout) t.layout_keys.push_back(sem_key(e.name, e.idx));
        std::sort(t.layout_keys.begin(), t.layout_keys.end());
        auto b = t.g.ps ? P.r->blobs.find(t.g.ps) : P.r->blobs.end();
        if (b != P.r->blobs.end()) target_sig(shader_info(b->second.first, b->second.second), &t.tgt_n, &t.tgt_types);
    }
}

static ID3D12RootSignature* get_root(Pre& P, uint64_t h) { return root_for(P.r, h, &P.roots); }   // or one seen this launch

// a root signature's flags, read from its serialized blob (RTS0 chunk); ~0 when unknown
static uint32_t root_flags(Pre& P, uint64_t h) {
    const uint8_t* b = nullptr; size_t n = 0;
    AcquireSRWLockShared(&P.r->lock);
    auto d = P.r->blobs.find(h);
    if (d != P.r->blobs.end()) { b = d->second.first; n = d->second.second; }
    ReleaseSRWLockShared(&P.r->lock);
    if (!b) { auto x = P.roots.find(h); if (x != P.roots.end()) { b = x->second.data(); n = x->second.size(); } }
    const uint8_t* p; uint32_t sz;
    if (!b || !dxbc_chunk(b, n, FOURCC4('R', 'T', 'S', '0'), &p, &sz) || sz < 24) return ~0u;
    uint32_t f; memcpy(&f, p + 20, 4); return f;
}

static uint32_t fmt_for(uint32_t ctype, uint8_t mask) {
    int n = 0; for (int m = mask; m; m >>= 1) n++;
    if (n < 1) n = 1;
    static const uint32_t F[4][4] = {{2, 2, 2, 2}, {42, 17, 7, 3}, {43, 18, 8, 4}, {41, 16, 6, 2}};   // DXGI R32[G32[B32[A32]]] uint/sint/float
    return F[ctype < 4 ? ctype : 3][n - 1];
}

// component type of a vertex format: 1 uint, 2 sint, 3 float (UNORM/SNORM/FLOAT read as float)
static uint32_t fmt_class(uint32_t f) {
    switch (f) {
    case 3: case 7: case 12: case 17: case 25: case 30: case 36: case 42: case 50: case 57: case 62: return 1;
    case 4: case 8: case 14: case 18: case 32: case 38: case 43: case 52: case 59: case 64: return 2;
    default: return 3;
    }
}

static void mark_done(Pre& P, uint64_t k) {
    inflight_note(0);   // built: no longer in flight
    AcquireSRWLockExclusive(&P.lock);
    if (P.done.insert(k).second && P.done_file) { fwrite(&k, 8, 1, P.done_file); fflush(P.done_file); }
    ReleaseSRWLockExclusive(&P.lock);
}

// every attempt for the current shader, for the log
static thread_local std::string t_why;
static std::string take_why() {
    std::string w; w.swap(t_why);
    if (w.size() > 16000) { w.resize(16000); w += " ..."; }
    return w;
}
// build VS+PS with template t's root signature and render state; input layout follows the VS
static HRESULT build_gfx(Pre& P, const Tmpl& t, const uint8_t* vs, uint32_t vsn, const ShaderInfo& vsi,
                         const uint8_t* ps, uint32_t psn,
                         const uint8_t* hs = nullptr, uint32_t hsn = 0, const uint8_t* ds = nullptr, uint32_t dsn = 0) {
    ID3D12RootSignature* rs = t.g.root ? get_root(P, t.g.root) : nullptr;
    if (t.g.root && !rs) return E_FAIL;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC d = {};
    d.pRootSignature = rs;
    d.VS = {vs, vsn}; if (ps) d.PS = {ps, psn};
    if (hs && ds) { d.HS = {hs, hsn}; d.DS = {ds, dsn}; }
    memcpy(&d.BlendState, t.g.blend, 328); d.SampleMask = t.g.sample_mask;
    memcpy(&d.RasterizerState, t.g.rast, 44); memcpy(&d.DepthStencilState, t.g.depth, 52);
    std::vector<D3D12_INPUT_ELEMENT_DESC> el;
    for (const SigEl& e : vsi.in) {
        if (e.sv != 0) continue;   // SV_VertexID etc. are not fed by the input assembler
        D3D12_INPUT_ELEMENT_DESC x = {e.name.c_str(), e.idx, (DXGI_FORMAT)fmt_for(e.ctype, e.mask), 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0};
        for (const Elem& m : t.g.layout)
            if (m.idx == e.idx && _stricmp(m.name.c_str(), e.name.c_str()) == 0) {
                // keep the game's format unless the VS reads a different component type (int vs float)
                if (fmt_class(m.fmt) == (e.ctype ? e.ctype : 3)) x.Format = (DXGI_FORMAT)m.fmt;
                x.InputSlot = m.slot; x.AlignedByteOffset = m.off;
                x.InputSlotClass = (D3D12_INPUT_CLASSIFICATION)m.cls; x.InstanceDataStepRate = m.step; break;
            }
        el.push_back(x);
    }
    d.InputLayout = {el.empty() ? nullptr : el.data(), (UINT)el.size()};
    d.IBStripCutValue = (D3D12_INDEX_BUFFER_STRIP_CUT_VALUE)t.g.strip;
    d.PrimitiveTopologyType = (D3D12_PRIMITIVE_TOPOLOGY_TYPE)t.g.topo;
    d.NumRenderTargets = ps ? t.g.nrt : 0;
    if (ps) memcpy(d.RTVFormats, t.g.rtv, 32);
    d.DSVFormat = (DXGI_FORMAT)t.g.dsv;
    memcpy(&d.SampleDesc, t.g.sd, 8); d.NodeMask = t.g.node; d.Flags = (D3D12_PIPELINE_STATE_FLAGS)t.g.flags;
    ID3D12PipelineState* pso = nullptr;
    HRESULT hr = o_CreateGraphicsPipelineState(P.r->dev, &d, IID_ID3D12PipelineState, (void**)&pso);
    if (pso) pso->Release();
    if (g_log) {   // which state was tried and how it went, e.g. "g12 ok" or "t0 0x80070057"
        bool tess = !P.tess.empty() && &t >= &P.tess[0] && &t <= &P.tess.back();
        char b[48]; snprintf(b, sizeof b, " %c%d %s", tess ? 't' : 'g', (int)(tess ? &t - &P.tess[0] : &t - &P.gfx[0]),
                             SUCCEEDED(hr) ? "ok" : "");
        t_why += b;
        if (FAILED(hr)) { snprintf(b, sizeof b, "0x%08lx", (unsigned long)hr); t_why += b; }
    }
    return hr;
}

struct Sh { const uint8_t* p; uint32_t n; uint64_t key; ShaderInfo info; };

// Last resort when no state the game used accepts a shader: a plain state made from the
// shaders' own signatures (opaque, no culling, depth test on, one render target per PS
// output). The driver keeps most of a shader's compiled code for other states, so this
// still saves most of the in-game compile.
static HRESULT build_synth(Pre& P, const Sh& vs, const Sh* ps, const Sh* hs, const Sh* ds, const Sh* gs) {
    std::vector<uint64_t> roots;
    for (auto* list : {&P.gfx, &P.tess}) for (const Tmpl& t : *list)
        if (std::find(roots.begin(), roots.end(), t.g.root) == roots.end()) roots.push_back(t.g.root);
    std::vector<D3D12_INPUT_ELEMENT_DESC> el;
    for (const SigEl& e : vs.info.in)
        if (e.sv == 0) el.push_back({e.name.c_str(), e.idx, (DXGI_FORMAT)fmt_for(e.ctype, e.mask), 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0});
    D3D12_GRAPHICS_PIPELINE_STATE_DESC d = {};
    d.VS = {vs.p, vs.n};
    if (ps) d.PS = {ps->p, ps->n};
    if (hs && ds) { d.HS = {hs->p, hs->n}; d.DS = {ds->p, ds->n}; }
    if (gs) d.GS = {gs->p, gs->n};
    for (int i = 0; i < 8; i++) d.BlendState.RenderTarget[i].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    d.BlendState.RenderTarget[0].SrcBlend = d.BlendState.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
    for (int i = 0; i < 8; i++) {
        auto& r = d.BlendState.RenderTarget[i];
        r.SrcBlend = r.SrcBlendAlpha = D3D12_BLEND_ONE; r.DestBlend = r.DestBlendAlpha = D3D12_BLEND_ZERO;
        r.BlendOp = r.BlendOpAlpha = D3D12_BLEND_OP_ADD; r.LogicOp = D3D12_LOGIC_OP_NOOP;
    }
    d.SampleMask = 0xFFFFFFFF;
    d.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID; d.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    d.RasterizerState.DepthClipEnable = TRUE;
    d.DepthStencilState.DepthEnable = TRUE; d.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    d.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_GREATER_EQUAL;
    d.DepthStencilState.FrontFace = d.DepthStencilState.BackFace = {D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_COMPARISON_FUNC_ALWAYS};
    d.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    if (ps) {
        for (const SigEl& e : ps->info.out)
            if (e.sv == 64 && e.reg < 8) {
                d.RTVFormats[e.reg] = e.ctype == 1 ? DXGI_FORMAT_R32G32B32A32_UINT : e.ctype == 2 ? DXGI_FORMAT_R32G32B32A32_SINT : DXGI_FORMAT_R16G16B16A16_FLOAT;
                if (e.reg + 1 > d.NumRenderTargets) d.NumRenderTargets = e.reg + 1;
            }
        for (UINT i = 0; i < d.NumRenderTargets; i++) if (!d.RTVFormats[i]) d.RTVFormats[i] = DXGI_FORMAT_R8G8B8A8_UNORM;
    }
    d.InputLayout = {el.empty() ? nullptr : el.data(), (UINT)el.size()};
    d.SampleDesc.Count = 1;
    // topology must match a geometry shader's input primitive, so try the kinds in turn
    std::vector<D3D12_PRIMITIVE_TOPOLOGY_TYPE> topo;
    if (hs && ds) topo = {D3D12_PRIMITIVE_TOPOLOGY_TYPE_PATCH};
    else if (gs) topo = {D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT, D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE};
    else topo = {D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE};
    HRESULT hr = E_FAIL;
    for (uint64_t root : roots) {
        uint32_t rf = root ? root_flags(P, root) : 0;
        if (!el.empty() && rf != ~0u && !(rf & 1)) continue;   // needs ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
        ID3D12RootSignature* rs = root ? get_root(P, root) : nullptr;
        if (root && !rs) continue;
        d.pRootSignature = rs;
        for (auto tp : topo) {
            d.PrimitiveTopologyType = tp;
            ID3D12PipelineState* pso = nullptr;
            hr = o_CreateGraphicsPipelineState(P.r->dev, &d, IID_ID3D12PipelineState, (void**)&pso);
            if (pso) pso->Release();
            if (g_log) {
                char b[64]; snprintf(b, sizeof b, " synth%s", SUCCEEDED(hr) ? " ok" : "");
                t_why += b;
                if (FAILED(hr)) { snprintf(b, sizeof b, " 0x%08lx", (unsigned long)hr); t_why += b; }
            }
            if (SUCCEEDED(hr)) { P.synth_built++; return hr; }
        }
    }
    return hr;
}

// after the best guesses fail: a few more of the game's states, then the plain state above
static const int EXTRA_TRIES = 16;

static int ps_max_target(const ShaderInfo& ps) {
    int m = -1;
    for (const SigEl& e : ps.out) if (e.sv == 64 /* TARGET */ && (int)e.reg > m) m = (int)e.reg;
    return m;
}

// G2S header: the container's name is at +0x10 (up to 32 chars); its family is the part before '-'
static std::string container_family(const std::vector<uint8_t>& d) {
    if (d.size() < 0x30) return std::string();
    if (!memcmp(d.data(), "_DOK0000", 8)) return "DOK";
    if (memcmp(d.data(), "_S2G", 4)) {   // some other entry type the deep scan found shaders in
        char t[16]; snprintf(t, sizeof t, "X_%c%c%c%c", isalnum(d[0]) ? d[0] : '_', isalnum(d[1]) ? d[1] : '_', isalnum(d[2]) ? d[2] : '_', isalnum(d[3]) ? d[3] : '_');
        return t;
    }
    const char* s = (const char*)d.data() + 0x10;
    std::string n(s, strnlen(s, 32));
    size_t dash = n.find('-');
    return dash == std::string::npos ? n : n.substr(0, dash);
}

// rank the game's render states for this VS+PS: the container's own state, then
// states the game used for the same shader family and the same PS outputs
static bool try_gfx(Pre& P, int own, const std::string& family, const Sh& vs, const Sh* ps) {
    int need = ps ? ps_max_target(ps->info) : -1, pn = 0; uint32_t pt = 0;
    if (ps) target_sig(ps->info, &pn, &pt);
    std::vector<uint64_t> vin;
    for (const SigEl& e : vs.info.in) if (e.sv == 0) vin.push_back(sem_key(e.name, e.idx));
    std::vector<std::pair<int, int>> cand;   // (score, template)
    for (int t = 0; t < (int)P.gfx.size(); t++) {
        const Tmpl& tm = P.gfx[t];
        if (ps && tm.max_target < need) continue;
        if (!ps && tm.g.ps) continue;
        int s = 0;
        if (t == own) s += 8;
        if (!family.empty() && tm.family == family) s += 4;
        if (ps && tm.tgt_n >= 0) { if (tm.tgt_n == pn && tm.tgt_types == pt) s += 4; }
        else if (ps && tm.max_target == need) s += 2;
        if (!ps && tm.g.nrt == 0) s += 2;
        bool all = true;
        for (uint64_t k : vin) if (!std::binary_search(tm.layout_keys.begin(), tm.layout_keys.end(), k)) { all = false; break; }
        if (all) s += 2;
        cand.push_back({s, t});
    }
    // higher score first; P.gfx is already ordered by how often the game uses each state
    std::stable_sort(cand.begin(), cand.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) { return a.first > b.first; });
    // build the best few states that the driver accepts: in a test on the game's own
    // pipelines, the real state was the top guess 49% of the time and in the top 2 71%
    int ok = 0; size_t i = 0;
    for (; i < cand.size() && i < (size_t)P.variants + 3 && ok < P.variants; i++) {
        const Tmpl& tm = P.gfx[cand[i].second];
        if (SUCCEEDED(build_gfx(P, tm, vs.p, vs.n, vs.info, ps ? ps->p : nullptr, ps ? ps->n : 0))) ok++;
    }
    for (size_t extra = 0; !ok && i < cand.size() && extra < EXTRA_TRIES; i++, extra++)
        if (SUCCEEDED(build_gfx(P, P.gfx[cand[i].second], vs.p, vs.n, vs.info, ps ? ps->p : nullptr, ps ? ps->n : 0))) ok++;
    if (!ok && SUCCEEDED(build_synth(P, vs, ps, nullptr, nullptr, nullptr))) ok++;
    return ok > 0;
}

// first pass over the archives: find which shader family each of the game's states belongs to
static void discover(Pre& P, const Parts& parts, const std::string& fam) {
    Replay* r = P.r;
    for (auto& pr : parts) {
        uint64_t k = shader_key(pr.first, pr.second);
        note_family(k, fam);
        if (r->wanted.count(k)) {   // a shader the seed's pipelines use: keep it, and record it for later launches
            AcquireSRWLockExclusive(&r->lock);
            if (!r->blobs.count(k)) {
                r->files.emplace_back(pr.first, pr.first + pr.second);
                r->blobs[k] = {r->files.back().data(), pr.second};
                r->adopted++;
                AcquireSRWLockExclusive(&g_lock); put_blob(BLOB_SHADER, pr.first, pr.second); ReleaseSRWLockExclusive(&g_lock);
            }
            ReleaseSRWLockExclusive(&r->lock);
        }
        bool dxil;
        if (shader_stage(pr.first, pr.second, &dxil) < 0) continue;
        AcquireSRWLockExclusive(&P.lock); P.discovered.insert(k); ReleaseSRWLockExclusive(&P.lock);
    }
}

static void learn_entry(Pre& P, const std::vector<uint8_t>& data) {
    std::string fam = container_family(data);
    Parts parts;
    split_dxbc(data.data(), data.size(), parts);
    discover(P, parts, fam);
    for (auto& pr : parts) {
        uint64_t k = shader_key(pr.first, pr.second);
        auto t = P.by_shader.find(k);
        if (t == P.by_shader.end()) continue;
        Tmpl& tm = P.gfx[t->second];
        ShaderInfo si = shader_info(pr.first, pr.second);
        AcquireSRWLockExclusive(&P.lock);
        if (!fam.empty()) tm.votes[fam]++;
        if (si.stage == 0 && k == tm.g.ps && tm.tgt_n < 0) target_sig(si, &tm.tgt_n, &tm.tgt_types);
        if (si.stage == 1 && k == tm.g.vs && !tm.vs_known) { tm.vs_info = si; tm.vs_known = true; }   // a seed state's VS
        ReleaseSRWLockExclusive(&P.lock);
    }
}

// tessellation: VS -> HS -> DS (-> PS) from one container, with a tessellation state the game used
static bool try_tess(Pre& P, const Sh& vs, const Sh& hs, const Sh& ds, const Sh* ps) {
    int need = ps ? ps_max_target(ps->info) : -1, pn = 0; uint32_t pt = 0;
    if (ps) target_sig(ps->info, &pn, &pt);
    std::vector<uint64_t> vin;
    for (const SigEl& e : vs.info.in) if (e.sv == 0) vin.push_back(sem_key(e.name, e.idx));
    int own = -1;
    for (uint64_t k : {ds.key, hs.key, vs.key}) { auto t = P.tess_by_shader.find(k); if (t != P.tess_by_shader.end()) { own = t->second; break; } }
    std::vector<std::pair<int, int>> cand;
    for (int t = 0; t < (int)P.tess.size(); t++) {
        const Tmpl& tm = P.tess[t];
        if (ps && tm.max_target < need) continue;
        if (!ps && tm.g.ps) continue;
        int s = t == own ? 8 : 0;
        if (ps && tm.tgt_n == pn && tm.tgt_types == pt) s += 4;
        if (!ps && tm.g.nrt == 0) s += 2;
        bool all = true;
        for (uint64_t k : vin) if (!std::binary_search(tm.layout_keys.begin(), tm.layout_keys.end(), k)) { all = false; break; }
        if (all) s += 2;
        cand.push_back({s, t});
    }
    std::stable_sort(cand.begin(), cand.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) { return a.first > b.first; });
    int ok = 0; size_t i = 0;
    for (; i < cand.size() && i < (size_t)P.variants + 3 && ok < P.variants; i++) {
        const Tmpl& tm = P.tess[cand[i].second];
        if (SUCCEEDED(build_gfx(P, tm, vs.p, vs.n, vs.info, ps ? ps->p : nullptr, ps ? ps->n : 0, hs.p, hs.n, ds.p, ds.n))) ok++;
    }
    if (!ok && SUCCEEDED(build_synth(P, vs, ps, &hs, &ds, nullptr))) ok++;
    return ok > 0;
}

// find VS -> HS -> DS in this container that can feed a stage reading ds_out-compatible inputs
static bool tess_chain(const std::vector<Sh>& vss, const std::vector<Sh>& hss, const Sh& ds, int* vi, int* hi) {
    for (size_t h = 0; h < hss.size(); h++) {
        if (!same_sig(hss[h].info.out, ds.info.in) || !same_sig(hss[h].info.pc, ds.info.pc)) continue;
        for (size_t v = 0; v < vss.size(); v++)
            if (linkable(vss[v].info.out, hss[h].info.in)) { *vi = (int)v; *hi = (int)h; return true; }
    }
    return false;
}

static void handle_shaders(Pre& P, const std::string& family, const std::vector<Sh>& shaders);

// every shader of one container or file: built, failed or skipped
static void handle_parts(Pre& P, const std::string& family, const Parts& parts) {
    std::vector<Sh> sm5, sm6;
    for (auto& pr : parts) {
        Sh s{pr.first, pr.second, shader_key(pr.first, pr.second), shader_info(pr.first, pr.second)};
        AcquireSRWLockExclusive(&P.lock); bool fresh = P.counted.insert(s.key).second; ReleaseSRWLockExclusive(&P.lock);
        if (fresh) P.processed++;
        (s.info.dxil ? sm6 : sm5).push_back(std::move(s));
    }
    // shader model 5 (DXBC) and 6 (DXIL) shaders can't be mixed in one pipeline
    if (!sm5.empty()) handle_shaders(P, family, sm5);
    if (!sm6.empty()) handle_shaders(P, family, sm6);
}

static void handle_entry(Pre& P, const std::vector<uint8_t>& data) {
    Parts parts;
    split_dxbc(data.data(), data.size(), parts);
    handle_parts(P, container_family(data), parts);
}

static void handle_shaders(Pre& P, const std::string& family, const std::vector<Sh>& shaders) {
    std::vector<Sh> vss, pss, css, hss, dss, gss;
    int own = -1; uint64_t own_cs_root = 0;
    for (const Sh& s : shaders) {
        auto t = P.by_shader.find(s.key);
        if (t != P.by_shader.end() && (own < 0 || s.info.stage == 0)) own = t->second;
        auto c = P.cs_root.find(s.key);
        if (c != P.cs_root.end()) own_cs_root = c->second;
        if (s.info.stage == 1) vss.push_back(s); else if (s.info.stage == 0) pss.push_back(s); else if (s.info.stage == 5) css.push_back(s);
        else if (s.info.stage == 3) hss.push_back(s); else if (s.info.stage == 4) dss.push_back(s); else if (s.info.stage == 2) gss.push_back(s);
    }
    const char* fam = family.c_str();
    auto K = [](uint64_t k) { return (unsigned long long)k; };
    auto first_time = [&](uint64_t k) {
        AcquireSRWLockExclusive(&P.lock); bool f = P.shader_seen.insert(k).second; ReleaseSRWLockExclusive(&P.lock);
        return f;
    };
    auto skip = [&](uint64_t k) {   // false: k is built next
        AcquireSRWLockShared(&P.lock); bool s = P.done.count(k) || P.recorded.count(k); ReleaseSRWLockShared(&P.lock);
        if (s) return true;
        if (P.bad.count(k)) return true;   // crashed the game twice
        if (P.lost_device || P.r->dev->GetDeviceRemovedReason() != S_OK) { P.lost_device = g_dev_lost = true; return true; }   // keep the crash note
        inflight_note(k);
        return false;
    };
    auto fail = [&]() {   // k could not be built: no longer in flight, unless it may have cost the device
        P.failed++;
        if (P.r->dev->GetDeviceRemovedReason() == S_OK) inflight_note(0);
    };
    // compute shaders
    for (const Sh& c : css) {
        uint64_t k = pair_key(c.key, 0);
        if (!first_time(c.key) || skip(k)) { P.skipped++; continue; }
        bool ok = false; int tries = 0; t_why.clear();
        std::vector<uint64_t> roots;
        if (own_cs_root) roots.push_back(own_cs_root);
        for (const CsTmpl& t : P.cs) if (t.root != own_cs_root) roots.push_back(t.root);
        for (uint64_t root : roots) {
            if (tries++ >= 4) break;
            ID3D12RootSignature* rs = get_root(P, root);
            char b[64];
            if (!rs) { snprintf(b, sizeof b, " root %016llx unavailable", K(root)); t_why += b; continue; }
            D3D12_COMPUTE_PIPELINE_STATE_DESC d = {}; d.pRootSignature = rs; d.CS = {c.p, c.n};
            ID3D12PipelineState* pso = nullptr;
            HRESULT hr = o_CreateComputePipelineState(P.r->dev, &d, IID_ID3D12PipelineState, (void**)&pso);
            if (pso) pso->Release();
            if (g_log) {
                snprintf(b, sizeof b, " root %016llx %s", K(root), SUCCEEDED(hr) ? "ok" : "");
                t_why += b;
                if (FAILED(hr)) { snprintf(b, sizeof b, "0x%08lx", (unsigned long)hr); t_why += b; }
            }
            if (SUCCEEDED(hr)) { ok = true; break; }
        }
        if (roots.empty()) t_why = " no compute root signature learned yet";
        if (ok) { P.built++; mark_done(P, k); logv("%s: built CS %016llx:%s", fam, K(c.key), take_why().c_str()); }
        else { fail(); logd("%s: CS %016llx failed: no root signature accepted it:%s", fam, K(c.key), take_why().c_str()); }
    }
    // one pipeline per pixel shader, with the first compatible VS (preferring one not used yet)
    std::vector<bool> vs_used(vss.size(), false);
    for (const Sh& p : pss) {
        if (!first_time(p.key)) { P.skipped++; continue; }
        int best = -1;
        for (int pass = 0; pass < 2 && best < 0; pass++)
            for (size_t i = 0; i < vss.size(); i++)
                if ((pass == 1 || !vs_used[i]) && linkable(vss[i].info.out, p.info.in)) { best = (int)i; break; }
        if (best >= 0) {
            vs_used[best] = true;
            first_time(vss[best].key);   // this VS is compiled with this PS
            uint64_t k = pair_key(vss[best].key, p.key);
            if (skip(k)) { P.skipped++; continue; }
            t_why.clear();
            if (try_gfx(P, own, family, vss[best], &p)) { P.built++; mark_done(P, k); logv("%s: built VS %016llx + PS %016llx:%s", fam, K(vss[best].key), K(p.key), take_why().c_str()); }
            else { fail(); logd("%s: VS %016llx + PS %016llx failed: no render state accepted:%s", fam, K(vss[best].key), K(p.key), take_why().c_str()); }
            continue;
        }
        // no VS fits: its input may come from a domain shader (tessellation)
        bool ok = false, tried_tess = false;
        t_why.clear();
        for (const Sh& ds : dss) {
            if (!linkable(ds.info.out, p.info.in)) continue;
            int vi, hi;
            if (!tess_chain(vss, hss, ds, &vi, &hi)) continue;
            tried_tess = true;
            uint64_t k = tess_key(vss[vi].key, hss[hi].key, ds.key, p.key);
            if (skip(k)) { ok = true; P.skipped++; break; }
            vs_used[vi] = true;
            if (try_tess(P, vss[vi], hss[hi], ds, &p)) {
                ok = true; P.built++; mark_done(P, k); P.tess_built++;
                logv("%s: built tessellation VS %016llx HS %016llx DS %016llx PS %016llx:%s", fam, K(vss[vi].key), K(hss[hi].key), K(ds.key), K(p.key), take_why().c_str());
                AcquireSRWLockExclusive(&P.lock); P.tess_seen.insert(ds.key); ReleaseSRWLockExclusive(&P.lock);
                break;
            }
        }
        if (ok) continue;
        if (tried_tess) { fail(); logd("%s: PS %016llx failed: no tessellation state accepted it:%s", fam, K(p.key), take_why().c_str()); continue; }
        // a geometry shader may feed it (e.g. it reads SV_RenderTargetArrayIndex): VS -> GS -> PS
        for (const Sh& g : gss) {
            if (!linkable(g.info.out, p.info.in)) continue;
            int vi = -1;
            for (size_t v = 0; v < vss.size() && vi < 0; v++) if (linkable(vss[v].info.out, g.info.in)) vi = (int)v;
            if (vi < 0) continue;
            uint64_t k = pair_key(pair_key(vss[vi].key, g.key), p.key) ^ 0x6E5;
            if (skip(k)) { ok = true; P.skipped++; break; }
            vs_used[vi] = true;
            AcquireSRWLockExclusive(&P.lock); P.gs_seen.insert(g.key); ReleaseSRWLockExclusive(&P.lock);
            if (SUCCEEDED(build_synth(P, vss[vi], &p, nullptr, nullptr, &g))) {
                ok = true; P.built++; P.gs_built++; mark_done(P, k);
                logv("%s: built VS %016llx + GS %016llx + PS %016llx:%s", fam, K(vss[vi].key), K(g.key), K(p.key), take_why().c_str());
                break;
            }
        }
        if (ok) continue;
        // no VS in this container fits: borrow a VS the game used with a compatible state
        for (int t = 0; t < (int)P.gfx.size() && !ok; t++) {
            Tmpl& tm = P.gfx[t];
            if (!tm.vs_known || tm.vs_info.dxil != p.info.dxil || tm.max_target < ps_max_target(p.info) || !linkable(tm.vs_info.out, p.info.in)) continue;
            auto b = P.r->blobs.find(tm.g.vs);
            if (b == P.r->blobs.end()) continue;
            uint64_t k = pair_key(tm.g.vs, p.key);
            if (skip(k)) { ok = true; P.skipped++; break; }
            if (SUCCEEDED(build_gfx(P, tm, b->second.first, b->second.second, tm.vs_info, p.p, p.n))) { ok = true; P.built++; mark_done(P, k); }
        }
        if (ok) logv("%s: built PS %016llx with a borrowed VS:%s", fam, K(p.key), take_why().c_str());
        else { fail(); logd("%s: PS %016llx failed: no VS or GS in its container fits and no borrowed VS worked:%s", fam, K(p.key), take_why().c_str()); }
    }
    // every domain shader not built above (its PS also had a plain VS, or it has no PS):
    // build it once with the first PS it can feed, or none (as for shadow passes)
    for (const Sh& ds : dss) {
        AcquireSRWLockShared(&P.lock); bool seen = P.tess_seen.count(ds.key) > 0; ReleaseSRWLockShared(&P.lock);
        if (seen) continue;
        int vi, hi;
        if (!tess_chain(vss, hss, ds, &vi, &hi)) {   // maybe another container has them
            logd("%s: DS %016llx skipped: no VS and HS in its container feed it", fam, K(ds.key));
            continue;
        }
        AcquireSRWLockExclusive(&P.lock); bool fresh = P.tess_seen.insert(ds.key).second; ReleaseSRWLockExclusive(&P.lock);
        if (!fresh) continue;
        vs_used[vi] = true;
        const Sh* ps = nullptr;
        for (const Sh& p : pss) if (linkable(ds.info.out, p.info.in)) { ps = &p; break; }
        uint64_t k = tess_key(vss[vi].key, hss[hi].key, ds.key, ps ? ps->key : 0);
        if (skip(k)) { P.skipped++; continue; }
        t_why.clear();
        if (try_tess(P, vss[vi], hss[hi], ds, ps) || (ps && try_tess(P, vss[vi], hss[hi], ds, nullptr))) {
            P.built++; mark_done(P, k); P.tess_built++;
            logv("%s: built DS %016llx (tessellation):%s", fam, K(ds.key), take_why().c_str());
        } else { fail(); logd("%s: DS %016llx failed: no tessellation state accepted it:%s", fam, K(ds.key), take_why().c_str()); }
    }
    // every geometry shader not built above: once, with a VS that feeds it and the first PS it feeds (or none)
    for (const Sh& g : gss) {
        AcquireSRWLockExclusive(&P.lock); bool fresh = P.gs_seen.insert(g.key).second; ReleaseSRWLockExclusive(&P.lock);
        if (!fresh) continue;
        int vi = -1;
        for (size_t v = 0; v < vss.size() && vi < 0; v++) if (linkable(vss[v].info.out, g.info.in)) vi = (int)v;
        if (vi < 0) { fail(); logd("%s: GS %016llx failed: no VS in its container feeds it", fam, K(g.key)); continue; }
        vs_used[vi] = true;
        const Sh* ps = nullptr;
        for (const Sh& p : pss) if (linkable(g.info.out, p.info.in)) { ps = &p; break; }
        uint64_t k = pair_key(pair_key(vss[vi].key, g.key), ps ? ps->key : 0) ^ 0x6E5;
        if (skip(k)) { P.skipped++; continue; }
        t_why.clear();
        if (SUCCEEDED(build_synth(P, vss[vi], ps, nullptr, nullptr, &g)) || (ps && SUCCEEDED(build_synth(P, vss[vi], nullptr, nullptr, nullptr, &g)))) {
            P.built++; P.gs_built++; mark_done(P, k);
            logv("%s: built GS %016llx:%s", fam, K(g.key), take_why().c_str());
        } else { fail(); logd("%s: GS %016llx failed:%s", fam, K(g.key), take_why().c_str()); }
    }
    // vertex shaders no pixel shader used: pair with the first compatible PS
    for (size_t i = 0; i < vss.size(); i++) {
        if (vs_used[i] || !first_time(vss[i].key)) continue;
        bool has_pos = false;
        for (const SigEl& e : vss[i].info.out) if (e.sv == 1) has_pos = true;
        if (!has_pos) {   // feeds a geometry/hull shader or stream output only: can't be drawn on its own
            P.no_position++; logv("%s: VS %016llx skipped: outputs no SV_Position, so it only runs ahead of a GS or HS", fam, K(vss[i].key));
            continue;
        }
        const Sh* ps = nullptr;
        for (const Sh& p : pss) if (linkable(vss[i].info.out, p.info.in)) { ps = &p; break; }
        uint64_t k = pair_key(vss[i].key, ps ? ps->key : 0);
        if (skip(k)) { P.skipped++; continue; }
        t_why.clear();
        if (try_gfx(P, own, family, vss[i], ps)) { P.built++; mark_done(P, k); logv("%s: built VS %016llx%s:%s", fam, K(vss[i].key), ps ? " with a PS" : " alone", take_why().c_str()); }
        else { fail(); logd("%s: VS %016llx failed: no render state accepted:%s", fam, K(vss[i].key), take_why().c_str()); }
    }
}

static DWORD WINAPI pre_worker(void* arg) {
    Pre& P = *(Pre*)arg;
    SetThreadPriority(GetCurrentThread(), g_cfg_pre_prio);
    std::vector<HANDLE> files(P.arcs.size(), INVALID_HANDLE_VALUE);
    std::vector<uint8_t> raw, data;
    size_t ng = P.learn ? 0 : P.groups.size();   // compile pass: the exe and other files first (each is one big group)
    t_slot = P.learn ? -1 : P.slots++ % 64;
    for (;;) {
        size_t i = P.next++;
        if (i >= ng + P.ents.size() || P.lost_device) break;
        if (P.r->dev->GetDeviceRemovedReason() != S_OK) { P.lost_device = g_dev_lost = true; break; }
        inflight_note(0);   // nothing is being built while the next container is read
        try {
            if (i < ng) {
                handle_parts(P, P.groups[i].first, P.groups[i].second);
                ov_set(OV_COMPILE, P.processed.load(), (long)P.discovered.size());
                continue;
            }
            size_t ei = i - ng;
            const G2SEntry& e = P.ents[ei];
            if (files[e.arc] == INVALID_HANDLE_VALUE)
                files[e.arc] = CreateFileW((package_dir() + P.arcs[e.arc].name).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
            if (files[e.arc] == INVALID_HANDLE_VALUE || e.esz > MAX_ENTRY || e.dsz > MAX_ENTRY) { P.bad_entries++; continue; }
            raw.resize((size_t)e.esz);
            LARGE_INTEGER li; li.QuadPart = (LONGLONG)e.off; DWORD rd = 0;
            SetFilePointerEx(files[e.arc], li, nullptr, FILE_BEGIN);
            if (!ReadFile(files[e.arc], raw.data(), (DWORD)e.esz, &rd, nullptr) || rd != e.esz) { P.bad_entries++; continue; }
            if (!unpack_entry(raw, e.esz, e.dsz, data)) { P.bad_entries++; continue; }
            if (P.learn) {
                learn_entry(P, data);
                ov_set(OV_DISCOVER, (long)ei + 1, (long)P.ents.size());
            } else {
                handle_entry(P, data);
                ov_set(OV_COMPILE, P.processed.load(), (long)P.discovered.size());
                if (g_log_progress > 0 && ei % (size_t)g_log_progress == 0)
                    logf("  pre-compile %zu/%zu containers: built=%ld failed=%ld skipped=%ld", ei, P.ents.size(), P.built.load(), P.failed.load(), P.skipped.load());
            }
            if (raw.capacity() > (64u << 20)) std::vector<uint8_t>().swap(raw);
            if (data.capacity() > (64u << 20)) std::vector<uint8_t>().swap(data);
        } catch (...) {   // out of memory: skip this one, keep going
            P.bad_entries++;
            std::vector<uint8_t>().swap(raw); std::vector<uint8_t>().swap(data);
        }
    }
    if (!P.lost_device && P.r->dev->GetDeviceRemovedReason() == S_OK) inflight_note(0);
    else P.lost_device = g_dev_lost = true;   // keep the pipeline it was on for the next start
    t_slot = -1;
    for (HANDLE h : files) if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
    return 0;
}

// shaders compiled into the game's exe itself (UI, post-processing, some compute); not in any archive
struct ExeShaders {
    HANDLE file = INVALID_HANDLE_VALUE, map = nullptr; const uint8_t* view = nullptr; size_t size = 0;
    std::vector<std::pair<const uint8_t*, uint32_t>> parts;
    bool open() {
        wchar_t exe[MAX_PATH]; GetModuleFileNameW(nullptr, exe, MAX_PATH);
        file = CreateFileW(exe, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        if (file == INVALID_HANDLE_VALUE) return false;
        LARGE_INTEGER sz; if (!GetFileSizeEx(file, &sz)) return false;
        size = (size_t)sz.QuadPart;
        map = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!map) return false;
        view = (const uint8_t*)MapViewOfFile(map, FILE_MAP_READ, 0, 0, 0);
        if (!view) return false;
        split_dxbc(view, size, parts);
        return true;
    }
    ~ExeShaders() {
        if (view) UnmapViewOfFile(view);
        if (map) CloseHandle(map);
        if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
    }
};

// Which GPU, driver and game folder the pre-compile's progress belongs to. Drivers keep their shader
// cache per GPU, driver version and exe path, and lose it when any of them changes.
static uint64_t cache_identity(ID3D12Device* dev) {
    Buf b;
    LUID luid = dev->GetAdapterLuid();
    IDXGIFactory4* f = nullptr; IDXGIAdapter1* a = nullptr;
    if (real_CreateDXGIFactory1 && SUCCEEDED(real_CreateDXGIFactory1(IID_IDXGIFactory4, (void**)&f)) && f) {
        DXGI_ADAPTER_DESC1 d;
        if (SUCCEEDED(f->EnumAdapterByLuid(luid, IID_IDXGIAdapter1, (void**)&a)) && a && SUCCEEDED(a->GetDesc1(&d))) {
            LARGE_INTEGER umd = {};
            a->CheckInterfaceSupport(IID_IDXGIDevice, &umd);   // the user-mode driver version, on every vendor
            b.u32(d.VendorId); b.u32(d.DeviceId); b.u32(d.SubSysId); b.u32(d.Revision); b.u64((uint64_t)umd.QuadPart);
            logf("GPU: %ls (vendor %04x, device %04x), driver %u.%u.%u.%u", d.Description, d.VendorId, d.DeviceId,
                 HIWORD(umd.HighPart), LOWORD(umd.HighPart), HIWORD(umd.LowPart), LOWORD(umd.LowPart));
        } else logf("GPU: could not identify the adapter");
        if (a) a->Release();
        f->Release();
    }
    wchar_t exe[MAX_PATH]; GetModuleFileNameW(nullptr, exe, MAX_PATH);
    for (wchar_t* c = exe; *c; c++) *c = towlower(*c);
    b.raw(exe, wcslen(exe) * 2);
    return fnv(b.v.data(), b.v.size());
}

static const uint64_t DONE_MAGIC = 0x32454E4F444F5350ULL;   // "PSODONE2"

static void precompile(Replay* r) {
    std::unique_ptr<Pre> holder(new Pre()); Pre* pp = holder.get(); Pre& P = *pp;
    P.r = r;
    ULONGLONG t0 = GetTickCount64();
    auto finish = [&](const char* why) {
        if (why) logf("pre-compile: %s", why);
        if (P.lost_device) logf("pre-compile: stopped: the GPU device was lost (driver reset or crash)");
        inflight_close();
        if (P.done_file) { fclose(P.done_file); P.done_file = nullptr; }
        if (P.built + P.failed > 0) ov_set(OV_DONE, (long)P.discovered.size(), (long)P.discovered.size());
        else ov_set(OV_DONE, 0, 0);
        AcquireSRWLockExclusive(&g_lock);   // in case it stopped before learning from them
        g_live_taken = true;
        std::vector<std::vector<uint8_t>>().swap(g_live_gfx); std::vector<std::vector<uint8_t>>().swap(g_live_cs); g_live_roots.clear();
        ReleaseSRWLockExclusive(&g_lock);
        log_pso_times("before the pre-compile finished");
        g_pre_finished = 1;
        logf("pre-compile done in %.0fs: built=%ld failed=%ld skipped(already done)=%ld unreadable containers=%ld",
             (GetTickCount64() - t0) / 1000.0, P.built.load(), P.failed.load(), P.skipped.load(), P.bad_entries.load());
    };

    // Progress: the pairs built on earlier launches, for this GPU, driver and game folder. Start over
    // when they changed, or when the driver's cache was cleared: most of this PC's recordings had to
    // be compiled again in the replay. Flag 1 = the last launch started over for that reason, so a
    // driver that never keeps its cache doesn't make every launch start over.
    wchar_t exe[MAX_PATH]; GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::wstring en(exe); en = en.substr(en.find_last_of(L"\\/") + 1);   // WILD HEARTS.exe and the trial exe
    for (wchar_t& c : en) if (c == L' ' || c == L'.') c = L'_';
    std::wstring done_name = g_dir + L"synth_done_" + en + L".bin";
    uint64_t id = cache_identity(r->dev);
    std::vector<uint8_t> f = read_file(done_name);
    uint64_t hdr[3] = {};
    if (f.size() >= 24) memcpy(hdr, f.data(), 24);
    bool same = hdr[0] == DONE_MAGIC && hdr[1] == id, lost = r->own_ok >= 200 && r->own_slow * 10 >= r->own_ok * 9;
    uint64_t flags = 0;
    bool fresh = !same;
    if (!same && hdr[0] == DONE_MAGIC) logf("pre-compile: the GPU, driver or game folder changed; building everything again");
    else if (same && lost && !(hdr[2] & 1)) { fresh = true; flags = 1; logf("pre-compile: the driver's shader cache was cleared; building everything again"); }
    else if (same && lost) { flags = 1; logf("pre-compile: the driver's shader cache lost this PC's pipelines again; not starting over a second time"); }
    if (!fresh) for (size_t i = 24; i + 8 <= f.size(); i += 8) { uint64_t k; memcpy(&k, &f[i], 8); P.done.insert(k); }
    P.done_file = _wfopen(done_name.c_str(), fresh ? L"wb" : L"r+b");
    if (P.done_file) {
        uint64_t h[3] = {DONE_MAGIC, id, flags};
        if (fresh) fwrite(h, 8, 3, P.done_file);
        else { _fseeki64(P.done_file, 16, SEEK_SET); fwrite(&flags, 8, 1, P.done_file); _fseeki64(P.done_file, (long long)(24 + (f.size() - 24) / 8 * 8), SEEK_SET); }
        fflush(P.done_file);
    }
    // crash guard: the pipelines being built when the game last crashed or lost the device, on this
    // GPU and driver. One caught twice is skipped from then on (once may be the game's own crash).
    swprintf(g_inflight_path, 2 * MAX_PATH, L"%lsinflight_%ls.bin", g_dir.c_str(), en.c_str());
    std::wstring skip_name = g_dir + L"synth_skip_" + en + L".bin";
    std::vector<uint8_t> crashed = read_file(g_inflight_path), sk = read_file(skip_name);
    DeleteFileW(g_inflight_path);
    auto ours = [&](const std::vector<uint8_t>& v) { return v.size() >= 16 && !memcmp(v.data(), &DONE_MAGIC, 8) && !memcmp(v.data() + 8, &id, 8); };
    bool sk_same = ours(sk);
    std::unordered_map<uint64_t, int> strikes;
    if (sk_same) for (size_t i = 16; i + 8 <= sk.size(); i += 8) { uint64_t k; memcpy(&k, &sk[i], 8); strikes[k]++; }
    int caught = 0;
    if (ours(crashed)) {
        if (FILE* sf = _wfopen(skip_name.c_str(), sk_same ? L"ab" : L"wb")) {
            if (!sk_same) { fwrite(&DONE_MAGIC, 8, 1, sf); fwrite(&id, 8, 1, sf); }
            std::unordered_set<uint64_t> now;
            for (size_t i = 16; i + 8 <= crashed.size(); i += 8) {
                uint64_t k; memcpy(&k, &crashed[i], 8);
                if (k && now.insert(k).second && strikes[k] < 2) { strikes[k]++; fwrite(&k, 8, 1, sf); caught++; }
            }
            fclose(sf);
        }
    } else if (!sk_same) DeleteFileW(skip_name.c_str());   // from another GPU or driver
    for (auto& kv : strikes) if (kv.second >= 2) P.bad.insert(kv.first);
    if (caught) logf("pre-compile: the game didn't close normally last time, while %d pipeline(s) were being built", caught);
    if (!P.bad.empty()) logf("pre-compile: skipping %zu pipeline(s): the game crashed twice while they were being built", P.bad.size());

    // where the shaders are: the archive index, or the deep scan on the first launch and after a game update
    ULONGLONG ts = GetTickCount64();
    ov_set(OV_INDEX, 0, 0);
    if (!load_index(P.arcs, P.ents)) {
        P.arcs.clear(); P.ents.clear();
        scan_archives(P.arcs, P.ents);
        save_index(P.arcs, P.ents);
    }
    logf("pre-compile: %zu shader containers in %zu archive(s) (%.1fs); %zu pipelines already built on earlier launches",
         P.ents.size(), P.arcs.size(), (GetTickCount64() - ts) / 1000.0, P.done.size());

    // Templates come from pipelines the game built. On the very first launch there are none on
    // disk yet, so wait for the game to build some (the seed's are used after 3 minutes).
    for (;;) {
        size_t live, live_cs;
        AcquireSRWLockShared(&g_lock); live = g_live_gfx.size(); live_cs = g_live_cs.size(); ReleaseSRWLockShared(&g_lock);
        size_t own = r->psos.size() + live, elapsed = (size_t)(GetTickCount64() - t0);
        ov_set(OV_LEARN, (long)std::min<size_t>(own, 300), 300);
        if (own >= 300 && r->psos.size() + live_cs >= 1) break;
        if (elapsed > 180000 && own + r->seed_psos.size() > 0) break;
        if (elapsed > 600000) { finish("the game built no pipelines to copy settings from; skipped"); return; }
        if (r->dev->GetDeviceRemovedReason() != S_OK) { P.lost_device = g_dev_lost = true; finish(nullptr); return; }
        Sleep(2000);
    }
    build_templates(P);
    logf("pre-compile: %zu render states and %zu compute root signatures learned from the game", P.gfx.size(), P.cs.size());
    if (P.gfx.empty()) { finish("no usable render state yet; will try again next launch"); return; }

    // shaders compiled into the game's exe (UI, post-processing, some compute), and the ones the deep
    // scan found outside the archive containers (other exes, DLLs, other files)
    ExeShaders exe_sh;
    if (exe_sh.open()) { P.groups.push_back({"EXE", exe_sh.parts}); discover(P, exe_sh.parts, "EXE"); }
    else logf("pre-compile: could not read the exe for its built-in shaders");
    std::vector<uint8_t> deep = read_file(g_dir + L"deep_shaders.bin");
    {
        Reader in{deep.data(), deep.size()};
        if (deep.size() >= 8 && in.u32() == 0x50454544) {   // "DEEP"
            uint32_t ng = in.u32();
            for (uint32_t g = 0; g < ng && !in.bad; g++) {
                uint32_t l = in.u32();
                const uint8_t* nm = in.raw(l);   // the file it was found in
                if (!nm) break;
                std::string where((const char*)nm, l);
                Parts parts;
                uint32_t nb = in.u32();
                for (uint32_t k = 0; k < nb && !in.bad; k++) {
                    uint32_t bl = in.u32(); const uint8_t* bp = in.raw(bl);
                    if (bp) parts.push_back({bp, bl});
                }
                discover(P, parts, where);
                P.groups.push_back({where, std::move(parts)});
            }
        }
    }
    size_t nsh = 0; for (auto& g : P.groups) nsh += g.second.size();
    logf("pre-compile: %zu shaders in the exe and other files, outside the archive containers", nsh);

    auto run = [&]() {
        P.next = 0;
        std::vector<HANDLE> th;
        start_threads(g_cfg_pre_threads, pre_worker, pp, th, P.learn ? "pre-compile (reading)" : "pre-compile");
        if (!th.empty()) WaitForMultipleObjects((DWORD)th.size(), th.data(), TRUE, INFINITE);
        for (HANDLE h : th) CloseHandle(h);
    };
    // first pass: which shader family each of the game's render states belongs to
    ts = GetTickCount64();
    P.learn = true; run(); P.learn = false;
    if (P.lost_device) { finish(nullptr); return; }
    int fams = 0, sigs = 0;
    for (Tmpl& t : P.gfx) {
        int best = 0;
        for (auto& kv : t.votes) if (kv.second > best) { best = kv.second; t.family = kv.first; }
        fams += !t.family.empty(); sigs += t.tgt_n >= 0;
        logd("state %zu: family=%s used %llu times, %u render targets, outputs=%d", (size_t)(&t - &P.gfx[0]),
             t.family.empty() ? "?" : t.family.c_str(), (unsigned long long)t.uses, t.g.nrt, t.tgt_n);
    }
    logf("pre-compile: %zu unique shaders found; matched %d of %zu render states to a shader family, %d to their PS outputs (%.1fs)",
         P.discovered.size(), fams, P.gfx.size(), sigs, (GetTickCount64() - ts) / 1000.0);
    if (!r->wanted.empty()) logf("seed: found %ld of the %zu shaders it was missing in the game files", r->adopted, r->wanted.size());
    if (g_cfg_replay && r->adopted && !r->missed.empty()) {   // the seed pipelines skipped at startup for lack of them
        std::vector<std::pair<uint32_t, std::pair<const uint8_t*, uint32_t>>> late;
        for (size_t i : r->missed) late.push_back(r->todo[i]);
        r->todo.swap(late); r->todo_own = 0;
        long ok0 = r->ok, missing0 = r->missing;
        run_replay(r, "seed replay");
        logf("seed replay: ok=%ld, still missing shaders=%ld", r->ok - ok0, r->missing - missing0);
    }
    // second pass: build everything
    ov_set(OV_COMPILE, 0, (long)P.discovered.size());
    g_inflight = CreateFileW(g_inflight_path, GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (g_inflight != INVALID_HANDLE_VALUE) { uint64_t h[2] = {DONE_MAGIC, id}; DWORD w; WriteFile(g_inflight, h, 16, &w, nullptr); }
    run();
    P.groups.clear();
    logf("pre-compile: %zu tessellation states learned from the game, %ld tessellation pipelines built", P.tess.size(), P.tess_built.load());
    logf("pre-compile: %ld geometry shader pipelines built; %ld pipelines built with the fallback state after every learned state failed",
         P.gs_built.load(), P.synth_built.load());
    if (P.no_position) logf("pre-compile: %ld vertex shaders skipped: they output no position, so they only run ahead of a geometry or hull shader", P.no_position.load());
    finish(nullptr);
}
