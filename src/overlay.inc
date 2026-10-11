// ------------------------------------------------------------------ progress overlay
// Included from whpso.cpp. Draws one line of text into the top-left corner of the
// game's own frames saying what the DLL is doing (learning, replaying, scanning,
// compiling, with progress), by hooking IDXGISwapChain::Present. Nothing is drawn
// once it is done. Turned off with [Overlay] Enabled=0 in WH_PSO.ini.

// progress shared with the replay / pre-compile threads
enum { OV_IDLE = 0, OV_LEARN, OV_REPLAY, OV_DISCOVER, OV_COMPILE, OV_DONE, OV_SCAN, OV_INDEX };
static std::atomic<int> g_ov_phase{OV_IDLE};
static std::atomic<long> g_ov_a{0}, g_ov_b{0};
static std::atomic<long long> g_ov_bytes{0};   // deep scan: bytes unpacked so far
static std::atomic<ULONGLONG> g_ov_done_at{0};
static bool g_ov_enabled = true;

static void ov_set(int phase, long a, long b) {
    if (phase == OV_DONE) g_ov_done_at = GetTickCount64();   // before the phase, so 'done' is never seen as expired
    g_ov_a = a; g_ov_b = b; g_ov_phase = phase;
}

// every phase is shown, so the overlay always says what the DLL is doing
static std::wstring ov_text() {
    wchar_t t[256];
    long a = g_ov_a, b = g_ov_b;
    if (a > b) a = b;
    int pct = b ? (int)(100.0 * a / b) : 0;
    switch (g_ov_phase.load()) {
    case OV_LEARN: swprintf(t, 256, L"Shader pre-compiler: learning render settings from the game (%ld/%ld pipelines)", a, b); break;
    case OV_REPLAY: swprintf(t, 256, L"Shader pre-compiler: rebuilding recorded pipelines %ld/%ld - %d%%", a, b, pct); break;
    case OV_INDEX: swprintf(t, 256, L"Shader pre-compiler: loading the archive index"); break;
    case OV_SCAN: swprintf(t, 256, L"Shader pre-compiler: deep scan of every game archive %ld/%ld files, %.1f GB unpacked - %d%%", a, b,
                           g_ov_bytes.load() / 1e9, pct); break;
    case OV_DISCOVER: swprintf(t, 256, L"Shader pre-compiler: reading shader containers %ld/%ld - %d%%", a, b, pct); break;
    case OV_COMPILE: swprintf(t, 256, L"Shader pre-compiler: compiling shaders %ld/%ld - %d%%", a, b, pct); break;
    case OV_DONE:
        if (GetTickCount64() - g_ov_done_at > 8000) return std::wstring();
        if (b) swprintf(t, 256, L"Shader pre-compiler: done (%ld shaders)", b);
        else swprintf(t, 256, L"Shader pre-compiler: done");
        break;
    default: return std::wstring();
    }
    return t;
}

// ---- D3D12 drawing state (only touched from the presenting thread)
typedef HRESULT(STDMETHODCALLTYPE* PFN_PRESENT)(IDXGISwapChain*, UINT, UINT);
typedef HRESULT(STDMETHODCALLTYPE* PFN_PRESENT1)(IDXGISwapChain1*, UINT, UINT, const DXGI_PRESENT_PARAMETERS*);
typedef HRESULT(STDMETHODCALLTYPE* PFN_SETCS)(IDXGISwapChain3*, DXGI_COLOR_SPACE_TYPE);
typedef HRESULT(STDMETHODCALLTYPE* PFN_CSC)(IDXGIFactory*, IUnknown*, DXGI_SWAP_CHAIN_DESC*, IDXGISwapChain**);
typedef HRESULT(STDMETHODCALLTYPE* PFN_CSCHWND)(IDXGIFactory2*, IUnknown*, HWND, const DXGI_SWAP_CHAIN_DESC1*,
                                                const DXGI_SWAP_CHAIN_FULLSCREEN_DESC*, IDXGIOutput*, IDXGISwapChain1**);
static PFN_PRESENT o_Present;
static PFN_PRESENT1 o_Present1;
static PFN_SETCS o_SetColorSpace1;
static PFN_CSC o_CreateSwapChain;
static PFN_CSCHWND o_CreateSwapChainForHwnd;

static const int OV_RING = 3, OV_TEXW = 2048, OV_TEXH = 96;
struct Overlay {
    bool ready = false, broken = false;
    ID3D12Device* dev = nullptr;
    ID3D12CommandQueue* queue = nullptr;
    IDXGISwapChain* sc = nullptr;
    ID3D12RootSignature* root = nullptr;
    ID3D12PipelineState* pso = nullptr; DXGI_FORMAT pso_fmt = DXGI_FORMAT_UNKNOWN;
    ID3D12DescriptorHeap* rtv_heap = nullptr; ID3D12DescriptorHeap* srv_heap = nullptr;
    ID3D12Resource* tex = nullptr;
    ID3D12Resource* upload[OV_RING] = {};
    uint8_t* upload_ptr[OV_RING] = {};
    ID3D12CommandAllocator* alloc[OV_RING] = {};
    ID3D12GraphicsCommandList* list = nullptr;
    ID3D12Fence* fence = nullptr; HANDLE ev = nullptr;
    UINT64 fence_val = 0, slot_val[OV_RING] = {};
    int slot = 0;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {};
    std::wstring last_text; int text_w = 0, text_h = 0, last_font = 0;
    bool tex_written = false;
    std::vector<uint8_t> vs_code, ps_code;
};
static Overlay g_ov;
static SRWLOCK g_ov_lock = SRWLOCK_INIT;
static std::atomic<bool> g_ov_retired{false};   // done: the overlay holds nothing of the game's any more
static std::atomic<bool> g_dev_lost{false};      // the game's GPU device was lost (driver reset or crash)
// every D3D12 swap chain made through a hooked factory, its queue (no reference held: a live swap
// chain keeps its queue alive) and color space. The overlay draws on whichever one presents.
struct OvChain { IDXGISwapChain* sc; ID3D12CommandQueue* queue; DXGI_COLOR_SPACE_TYPE cs; };
static std::vector<OvChain> g_ov_chains;

// s: brightness of the white text, so it isn't blinding in HDR10 or dim in scRGB
static const char* OV_HLSL =
    "cbuffer C : register(b0) { float4 r; float2 vp; float2 org; float s; };\n"
    "Texture2D<float4> T : register(t0);\n"
    "float4 vs(uint id : SV_VertexID) : SV_Position {\n"
    "  float2 u = float2(id & 1, id >> 1);\n"
    "  float2 p = r.xy + u * r.zw;\n"
    "  return float4(p.x / vp.x * 2 - 1, 1 - p.y / vp.y * 2, 0, 1);\n"
    "}\n"
    "float4 ps(float4 pos : SV_Position) : SV_Target {\n"
    "  return T.Load(int3(int2(pos.xy - org), 0)) * float4(s, s, s, 1);\n"
    "}\n";

typedef HRESULT(WINAPI* PFN_SER_RS)(const D3D12_ROOT_SIGNATURE_DESC*, D3D_ROOT_SIGNATURE_VERSION, ID3DBlob**, ID3DBlob**);
typedef HRESULT(WINAPI* PFN_D3DCOMPILE)(const void*, SIZE_T, const char*, const void*, void*, const char*, const char*, UINT, UINT, ID3DBlob**, ID3DBlob**);

static bool ov_compile(std::vector<uint8_t>& out, const char* entry, const char* target) {
    HMODULE m = LoadLibraryW(L"d3dcompiler_47.dll");   // part of Windows 10 and 11
    PFN_D3DCOMPILE compile = m ? (PFN_D3DCOMPILE)GetProcAddress(m, "D3DCompile") : nullptr;
    if (!compile) return false;
    ID3DBlob* code = nullptr; ID3DBlob* err = nullptr;
    HRESULT hr = compile(OV_HLSL, strlen(OV_HLSL), "overlay", nullptr, nullptr, entry, target, 0, 0, &code, &err);
    if (err) { logf("overlay shader: %s", (const char*)err->GetBufferPointer()); err->Release(); }
    if (FAILED(hr) || !code) return false;
    out.assign((const uint8_t*)code->GetBufferPointer(), (const uint8_t*)code->GetBufferPointer() + code->GetBufferSize());
    code->Release();
    return true;
}

// compile the two overlay shaders on a background thread, before there is anything to show,
// so the game's present thread never waits on the compiler
static void ov_prepare() {
    if (!g_ov_enabled) return;
    std::vector<uint8_t> vs, ps;
    bool ok = ov_compile(vs, "vs", "vs_5_0") && ov_compile(ps, "ps", "ps_5_0");
    AcquireSRWLockExclusive(&g_ov_lock);
    if (ok) { g_ov.vs_code.swap(vs); g_ov.ps_code.swap(ps); }
    else { g_ov.broken = true; logf("overlay: could not compile its shaders, overlay disabled"); }
    ReleaseSRWLockExclusive(&g_ov_lock);
}

static bool ov_init(Overlay& o) {
    ID3D12Device* dev = o.dev;
    if (o.vs_code.empty() || o.ps_code.empty()) return false;
    // root signature: 9 constants (b0) + one SRV table (t0)
    D3D12_DESCRIPTOR_RANGE range = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
    D3D12_ROOT_PARAMETER params[2] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.Num32BitValues = 9; params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 1; params[1].DescriptorTable.pDescriptorRanges = &range;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_ROOT_SIGNATURE_DESC rd = {2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    HMODULE d3d = GetModuleHandleW(L"d3d12.dll");
    auto ser = d3d ? (PFN_SER_RS)GetProcAddress(d3d, "D3D12SerializeRootSignature") : nullptr;
    ID3DBlob* blob = nullptr; ID3DBlob* err = nullptr;
    if (!ser || FAILED(ser(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err))) { if (err) err->Release(); return false; }
    HRESULT hr = o_CreateRootSignature(dev, 0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_ID3D12RootSignature, (void**)&o.root);
    blob->Release();
    if (FAILED(hr)) return false;

    D3D12_DESCRIPTOR_HEAP_DESC hd = {D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    if (FAILED(dev->CreateDescriptorHeap(&hd, IID_ID3D12DescriptorHeap, (void**)&o.rtv_heap))) return false;
    hd = {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    if (FAILED(dev->CreateDescriptorHeap(&hd, IID_ID3D12DescriptorHeap, (void**)&o.srv_heap))) return false;

    D3D12_HEAP_PROPERTIES hp = {D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0};
    D3D12_RESOURCE_DESC td = {};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; td.Width = OV_TEXW; td.Height = OV_TEXH;
    td.DepthOrArraySize = 1; td.MipLevels = 1; td.Format = DXGI_FORMAT_R8G8B8A8_UNORM; td.SampleDesc.Count = 1;
    td.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr,
                                            IID_ID3D12Resource, (void**)&o.tex))) return false;
    UINT64 total = 0;
    dev->GetCopyableFootprints(&td, 0, 1, 0, &o.fp, nullptr, nullptr, &total);
    D3D12_SHADER_RESOURCE_VIEW_DESC sv = {};
    sv.Format = DXGI_FORMAT_R8G8B8A8_UNORM; sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; sv.Texture2D.MipLevels = 1;
    dev->CreateShaderResourceView(o.tex, &sv, o.srv_heap->GetCPUDescriptorHandleForHeapStart());

    hp.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC bd = {};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = total; bd.Height = 1; bd.DepthOrArraySize = 1;
    bd.MipLevels = 1; bd.SampleDesc.Count = 1; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    for (int i = 0; i < OV_RING; i++) {
        if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                IID_ID3D12Resource, (void**)&o.upload[i]))) return false;
        D3D12_RANGE none = {0, 0};
        if (FAILED(o.upload[i]->Map(0, &none, (void**)&o.upload_ptr[i]))) return false;
        if (FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_ID3D12CommandAllocator, (void**)&o.alloc[i]))) return false;
    }
    if (FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, o.alloc[0], nullptr, IID_ID3D12GraphicsCommandList, (void**)&o.list))) return false;
    o.list->Close();
    if (FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_ID3D12Fence, (void**)&o.fence))) return false;
    o.ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    return true;
}

static bool ov_pso(Overlay& o, DXGI_FORMAT fmt) {
    if (o.pso && o.pso_fmt == fmt) return true;
    if (o.pso) { o.pso->Release(); o.pso = nullptr; }
    D3D12_GRAPHICS_PIPELINE_STATE_DESC d = {};
    d.pRootSignature = o.root;
    d.VS = {o.vs_code.data(), o.vs_code.size()}; d.PS = {o.ps_code.data(), o.ps_code.size()};
    D3D12_RENDER_TARGET_BLEND_DESC& b = d.BlendState.RenderTarget[0];
    b.BlendEnable = TRUE; b.SrcBlend = D3D12_BLEND_ONE; b.DestBlend = D3D12_BLEND_INV_SRC_ALPHA; b.BlendOp = D3D12_BLEND_OP_ADD;
    b.SrcBlendAlpha = D3D12_BLEND_ONE; b.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA; b.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    b.LogicOp = D3D12_LOGIC_OP_NOOP; b.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    d.SampleMask = 0xFFFFFFFF;
    d.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID; d.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    d.RasterizerState.DepthClipEnable = TRUE;
    d.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    d.NumRenderTargets = 1; d.RTVFormats[0] = fmt; d.SampleDesc.Count = 1;
    if (FAILED(o_CreateGraphicsPipelineState(o.dev, &d, IID_ID3D12PipelineState, (void**)&o.pso))) return false;
    o.pso_fmt = fmt;
    return true;
}

// render the text with GDI into the given upload buffer (premultiplied RGBA)
static void ov_render_text(Overlay& o, const std::wstring& text, int font_px, uint8_t* dst) {
    HDC dc = CreateCompatibleDC(nullptr);
    BITMAPINFO bi = {};
    bi.bmiHeader.biSize = sizeof bi.bmiHeader; bi.bmiHeader.biWidth = OV_TEXW; bi.bmiHeader.biHeight = -OV_TEXH;
    bi.bmiHeader.biPlanes = 1; bi.bmiHeader.biBitCount = 32; bi.bmiHeader.biCompression = BI_RGB;
    void* bits = nullptr;
    HBITMAP bmp = CreateDIBSection(dc, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);
    HGDIOBJ old_bmp = SelectObject(dc, bmp);
    // the font shrinks until the longest line the overlay shows fits the texture (4K and up)
    static const wchar_t LONGEST[] = L"Shader pre-compiler: deep scan of every game archive 99999/99999 files, 999.9 GB unpacked - 100%";
    HFONT font = nullptr; HGDIOBJ old_font = nullptr; SIZE sz;
    for (;;) {
        font = CreateFontW(-font_px, 0, 0, 0, FW_SEMIBOLD, 0, 0, 0, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                           ANTIALIASED_QUALITY, DEFAULT_PITCH, L"Segoe UI");
        HGDIOBJ prev = SelectObject(dc, font);
        if (!old_font) old_font = prev;
        GetTextExtentPoint32W(dc, LONGEST, (int)wcslen(LONGEST), &sz);
        if ((sz.cx + font_px <= OV_TEXW && sz.cy + font_px / 2 <= OV_TEXH) || font_px <= 12) break;
        SelectObject(dc, old_font); DeleteObject(font);
        font_px = font_px * 9 / 10;
    }
    GetTextExtentPoint32W(dc, text.c_str(), (int)text.size(), &sz);
    int pad = font_px / 2;
    o.text_w = std::min<int>(OV_TEXW, sz.cx + 2 * pad); o.text_h = std::min<int>(OV_TEXH, sz.cy + pad);
    memset(bits, 0, OV_TEXW * OV_TEXH * 4);
    SetBkMode(dc, TRANSPARENT); SetTextColor(dc, RGB(255, 255, 255));
    TextOutW(dc, pad, pad / 2, text.c_str(), (int)text.size());
    GdiFlush();
    const uint8_t* src = (const uint8_t*)bits;
    for (int y = 0; y < OV_TEXH; y++) {
        uint8_t* row = dst + (size_t)y * o.fp.Footprint.RowPitch;
        for (int x = 0; x < OV_TEXW; x++) {
            const uint8_t* s = src + ((size_t)y * OV_TEXW + x) * 4;
            uint8_t* p = row + x * 4;
            if (x >= o.text_w || y >= o.text_h) { p[0] = p[1] = p[2] = p[3] = 0; continue; }
            int cov = std::max(s[0], std::max(s[1], s[2]));       // text coverage 0..255
            int a = 150 + (105 * cov) / 255;                       // dark translucent panel under white text
            p[0] = p[1] = p[2] = (uint8_t)cov; p[3] = (uint8_t)a;
        }
    }
    SelectObject(dc, old_font); DeleteObject(font);
    SelectObject(dc, old_bmp); DeleteObject(bmp); DeleteDC(dc);
}

// caller holds g_ov_lock. Frees everything made on o.dev once its GPU work is done.
static void ov_release(Overlay& o) {
    if (o.fence && o.ev && o.fence->GetCompletedValue() < o.fence_val) { o.fence->SetEventOnCompletion(o.fence_val, o.ev); WaitForSingleObject(o.ev, 2000); }
    IUnknown* objs[] = {o.pso, o.root, o.rtv_heap, o.srv_heap, o.tex, o.upload[0], o.upload[1], o.upload[2],
                        o.alloc[0], o.alloc[1], o.alloc[2], o.list, o.fence};
    static_assert(OV_RING == 3, "objs lists each ring slot");
    for (IUnknown* u : objs) if (u) u->Release();
    if (o.ev) CloseHandle(o.ev);
    std::vector<uint8_t> vs, ps; vs.swap(o.vs_code); ps.swap(o.ps_code);
    ID3D12Device* dev = o.dev; ID3D12CommandQueue* q = o.queue; IDXGISwapChain* sc = o.sc; bool broken = o.broken;
    o.~Overlay(); new (&o) Overlay();
    o.vs_code.swap(vs); o.ps_code.swap(ps); o.dev = dev; o.queue = q; o.sc = sc; o.broken = broken;
}

// caller holds g_ov_lock. Draw on this swap chain from now on; rebuild if its device is another one.
// let go of the game's device and queue, so the game can create a new device after a driver reset
static void ov_drop(Overlay& o) {
    ov_release(o);
    if (o.queue) o.queue->Release();
    if (o.dev) o.dev->Release();
    o.queue = nullptr; o.dev = nullptr; o.sc = nullptr;
}
static void ov_target(Overlay& o, IDXGISwapChain* sc, ID3D12CommandQueue* q) {
    if (g_ov_retired) return;
    if (q == o.queue) { o.sc = sc; return; }
    ID3D12Device* dev = nullptr;
    if (FAILED(q->GetDevice(IID_ID3D12Device, (void**)&dev))) dev = nullptr;
    if (dev && dev->GetDeviceRemovedReason() != S_OK) { dev->Release(); return; }   // a swap chain of a lost device
    o.sc = sc;
    if (dev != o.dev) {
        ov_release(o);
        if (o.dev) o.dev->Release();
        o.dev = dev;
    } else if (dev) dev->Release();
    if (o.queue) o.queue->Release();
    q->AddRef(); o.queue = q;
}

static void ov_draw(IDXGISwapChain* sc) {
    std::wstring text = g_ov_enabled ? ov_text() : std::wstring();
    if (text.empty()) {
        if (g_ov_phase == OV_DONE && !g_ov_retired && TryAcquireSRWLockExclusive(&g_ov_lock)) {   // finished for good
            g_ov_retired = true; ov_drop(g_ov);
            ReleaseSRWLockExclusive(&g_ov_lock);
        }
        return;
    }
    if (!TryAcquireSRWLockExclusive(&g_ov_lock)) return;
    Overlay& o = g_ov;
    DXGI_COLOR_SPACE_TYPE cs = DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
    bool known = false;
    for (const OvChain& c : g_ov_chains)
        if (c.sc == sc) { known = true; cs = c.cs; if (sc != o.sc) ov_target(o, sc, c.queue); break; }
    if (o.broken || !known || !o.dev || sc != o.sc || !o_CreateGraphicsPipelineState || !o_CreateRootSignature) { ReleaseSRWLockExclusive(&g_ov_lock); return; }
    if (!o.ready) {
        if (!ov_init(o)) { o.broken = true; logf("overlay: setup failed, overlay disabled"); ReleaseSRWLockExclusive(&g_ov_lock); return; }
        o.ready = true;
    }
    IDXGISwapChain3* sc3 = nullptr;
    DXGI_SWAP_CHAIN_DESC desc;
    if (FAILED(sc->QueryInterface(IID_IDXGISwapChain3, (void**)&sc3)) || FAILED(sc->GetDesc(&desc))) {
        if (sc3) sc3->Release();
        ReleaseSRWLockExclusive(&g_ov_lock); return;
    }
    UINT idx = sc3->GetCurrentBackBufferIndex(); sc3->Release();
    ID3D12Resource* bb = nullptr;
    if (FAILED(sc->GetBuffer(idx, IID_ID3D12Resource, (void**)&bb)) || !ov_pso(o, desc.BufferDesc.Format)) {
        if (bb) bb->Release();
        ReleaseSRWLockExclusive(&g_ov_lock); return;
    }
    // wait until this ring slot's previous work is done
    int s = o.slot; o.slot = (o.slot + 1) % OV_RING;
    if (o.fence->GetCompletedValue() < o.slot_val[s]) { o.fence->SetEventOnCompletion(o.slot_val[s], o.ev); WaitForSingleObject(o.ev, 2000); }
    if (o.fence->GetCompletedValue() < o.slot_val[s]) {   // still busy: skip this frame rather than reuse a busy allocator
        o.slot = s; bb->Release(); ReleaseSRWLockExclusive(&g_ov_lock); return;
    }
    UINT H = desc.BufferDesc.Height ? desc.BufferDesc.Height : 1080, W = desc.BufferDesc.Width ? desc.BufferDesc.Width : 1920;
    int font_px = std::max<int>(14, (int)H / 48);
    bool upload = text != o.last_text || font_px != o.last_font || !o.tex_written;
    if (upload) { ov_render_text(o, text, font_px, o.upload_ptr[s]); o.last_text = text; o.last_font = font_px; o.tex_written = true; }

    o.alloc[s]->Reset();
    o.list->Reset(o.alloc[s], o.pso);
    D3D12_RESOURCE_BARRIER br[2] = {};
    br[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; br[0].Transition.pResource = bb;
    br[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    br[0].Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT; br[0].Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    int nb = 1;
    if (upload) {
        br[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; br[1].Transition.pResource = o.tex;
        br[1].Transition.Subresource = 0;
        br[1].Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE; br[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        nb = 2;
    }
    o.list->ResourceBarrier(nb, br);
    if (upload) {
        D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
        dst.pResource = o.tex; dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; dst.SubresourceIndex = 0;
        src.pResource = o.upload[s]; src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; src.PlacedFootprint = o.fp;
        o.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        std::swap(br[1].Transition.StateBefore, br[1].Transition.StateAfter);
        o.list->ResourceBarrier(1, &br[1]);
    }
    D3D12_CPU_DESCRIPTOR_HANDLE rtv = o.rtv_heap->GetCPUDescriptorHandleForHeapStart();
    o.dev->CreateRenderTargetView(bb, nullptr, rtv);
    o.list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    D3D12_VIEWPORT vp = {0, 0, (float)W, (float)H, 0, 1};
    D3D12_RECT sr = {0, 0, (LONG)W, (LONG)H};
    o.list->RSSetViewports(1, &vp); o.list->RSSetScissorRects(1, &sr);
    o.list->SetGraphicsRootSignature(o.root);
    ID3D12DescriptorHeap* heaps[1] = {o.srv_heap};
    o.list->SetDescriptorHeaps(1, heaps);
    float margin = (float)font_px;
    // white is 1.0: right for SDR, about 10,000 nits in HDR10 (PQ) and 80 nits in scRGB
    float bright = cs == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020 ? 0.56f
                 : (cs == DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709 || desc.BufferDesc.Format == DXGI_FORMAT_R16G16B16A16_FLOAT) ? 2.5f : 1.0f;
    float c[9] = {margin, margin, (float)o.text_w, (float)o.text_h, (float)W, (float)H, margin, margin, bright};
    o.list->SetGraphicsRoot32BitConstants(0, 9, c, 0);
    o.list->SetGraphicsRootDescriptorTable(1, o.srv_heap->GetGPUDescriptorHandleForHeapStart());
    o.list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    o.list->DrawInstanced(4, 1, 0, 0);
    std::swap(br[0].Transition.StateBefore, br[0].Transition.StateAfter);
    o.list->ResourceBarrier(1, &br[0]);
    if (SUCCEEDED(o.list->Close())) {
        ID3D12CommandList* lists[1] = {o.list};
        o.queue->ExecuteCommandLists(1, lists);
        o.queue->Signal(o.fence, ++o.fence_val);
        o.slot_val[s] = o.fence_val;
    }
    bb->Release();
    ReleaseSRWLockExclusive(&g_ov_lock);
}

static void ov_after_present(HRESULT hr) {
    if (hr != DXGI_ERROR_DEVICE_REMOVED && hr != DXGI_ERROR_DEVICE_RESET) return;
    g_dev_lost = true;
    AcquireSRWLockExclusive(&g_ov_lock);
    if (g_ov.dev) { ov_drop(g_ov); logf("overlay: the GPU device was lost; released it"); }
    ReleaseSRWLockExclusive(&g_ov_lock);
}
static HRESULT STDMETHODCALLTYPE h_Present(IDXGISwapChain* sc, UINT sync, UINT flags) {
    if (!(flags & DXGI_PRESENT_TEST)) ov_draw(sc);
    HRESULT hr = o_Present(sc, sync, flags);
    ov_after_present(hr);
    return hr;
}
static HRESULT STDMETHODCALLTYPE h_Present1(IDXGISwapChain1* sc, UINT sync, UINT flags, const DXGI_PRESENT_PARAMETERS* p) {
    if (!(flags & DXGI_PRESENT_TEST)) ov_draw(sc);
    HRESULT hr = o_Present1(sc, sync, flags, p);
    ov_after_present(hr);
    return hr;
}
static HRESULT STDMETHODCALLTYPE h_SetColorSpace1(IDXGISwapChain3* sc, DXGI_COLOR_SPACE_TYPE cs) {
    HRESULT hr = o_SetColorSpace1(sc, cs);
    if (SUCCEEDED(hr)) {
        AcquireSRWLockExclusive(&g_ov_lock);
        for (OvChain& c : g_ov_chains) if (c.sc == sc) c.cs = cs;
        ReleaseSRWLockExclusive(&g_ov_lock);
    }
    return hr;
}

// remember every D3D12 swap chain and the queue it presents on, and hook Present (one vtable for all)
static void ov_track(IUnknown* device, IDXGISwapChain* sc) {
    if (!sc || !device) return;
    ID3D12CommandQueue* q = nullptr;
    if (FAILED(device->QueryInterface(IID_ID3D12CommandQueue, (void**)&q)) || !q) return;   // not a D3D12 swap chain
    AcquireSRWLockExclusive(&g_ov_lock);
    bool found = false;
    for (OvChain& c : g_ov_chains) if (c.sc == sc) { c = {sc, q, DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709}; found = true; }
    if (!found) {
        if (g_ov_chains.size() >= 16) g_ov_chains.erase(g_ov_chains.begin());
        g_ov_chains.push_back({sc, q, DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709});
    }
    ov_target(g_ov, sc, q);   // the newest swap chain until another one presents
    void** vt = *(void***)sc;
    if (!o_Present) {
        patch(vt, 8, (void*)h_Present, (void**)&o_Present);
        patch(vt, 22, (void*)h_Present1, (void**)&o_Present1);
        IDXGISwapChain3* sc3 = nullptr;
        if (SUCCEEDED(sc->QueryInterface(IID_IDXGISwapChain3, (void**)&sc3)) && sc3) {
            patch(vt, 38, (void*)h_SetColorSpace1, (void**)&o_SetColorSpace1);
            sc3->Release();
        }
        logf("overlay: hooked the game's swap chain");
    }
    ReleaseSRWLockExclusive(&g_ov_lock);
    q->Release();
}

static HRESULT STDMETHODCALLTYPE h_CreateSwapChain(IDXGIFactory* f, IUnknown* dev, DXGI_SWAP_CHAIN_DESC* d, IDXGISwapChain** out) {
    HRESULT hr = o_CreateSwapChain(f, dev, d, out);
    if (SUCCEEDED(hr) && out) ov_track(dev, *out);
    return hr;
}
static HRESULT STDMETHODCALLTYPE h_CreateSwapChainForHwnd(IDXGIFactory2* f, IUnknown* dev, HWND w, const DXGI_SWAP_CHAIN_DESC1* d,
                                                         const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* fs, IDXGIOutput* o, IDXGISwapChain1** out) {
    HRESULT hr = o_CreateSwapChainForHwnd(f, dev, w, d, fs, o, out);
    if (SUCCEEDED(hr) && out) ov_track(dev, *out);
    return hr;
}

// called on every factory the game creates
static void ov_hook_factory(void* factory) {
    if (!g_ov_enabled || !factory) return;
    static std::atomic<int> once{0};
    if (once.exchange(1)) return;   // all factories share one vtable
    void** vt = *(void***)factory;
    patch(vt, 10, (void*)h_CreateSwapChain, (void**)&o_CreateSwapChain);
    IDXGIFactory2* f2 = nullptr;
    if (SUCCEEDED(((IUnknown*)factory)->QueryInterface(IID_IDXGIFactory2, (void**)&f2)) && f2) {
        patch(*(void***)f2, 15, (void*)h_CreateSwapChainForHwnd, (void**)&o_CreateSwapChainForHwnd);
        f2->Release();
    }
}
