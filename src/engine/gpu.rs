//! SOTA Zero-Dependency OpenCL GPU Compute Engine for Number-Theoretic Transform (NTT)
//! Dynamically binds to `OpenCL.dll` on Windows (`LoadLibraryA` / `GetProcAddress`) and
//! `libOpenCL.so.1` / `OpenCL.framework` on Linux/Unix (`dlopen` / `dlsym`), preferring
//! high-compute discrete GPUs with persistent VRAM buffer retainment and 0.0% idle standby.

use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

#[inline(always)]
const fn pow_mod_u32(base: u32, mut exp: u32, m: u32) -> u32 {
    let mut res = 1u64;
    let mut b = (base % m) as u64;
    let m64 = m as u64;
    while exp > 0 {
        if (exp & 1) != 0 {
            res = (res * b) % m64;
        }
        b = (b * b) % m64;
        exp >>= 1;
    }
    res as u32
}

#[cfg(any(windows, unix))]
mod opencl_ffi {
    use super::*;
    #[cfg(windows)]
    use windows::core::PCSTR;
    #[cfg(windows)]
    use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};

    pub const CL_SUCCESS: i32 = 0;
    pub const CL_DEVICE_TYPE_GPU: u64 = 1 << 2;
    pub const CL_DEVICE_NAME: u32 = 0x102B;
    pub const CL_DEVICE_VENDOR_ID: u32 = 0x1001;
    pub const CL_DEVICE_MAX_COMPUTE_UNITS: u32 = 0x1002;
    pub const CL_MEM_READ_WRITE: u64 = 1 << 0;
    pub const CL_TRUE: u32 = 1;
    pub const CL_FALSE: u32 = 0;

    pub type ClGetPlatformIDs = unsafe extern "system" fn(u32, *mut *mut c_void, *mut u32) -> i32;
    pub type ClGetDeviceIDs =
        unsafe extern "system" fn(*mut c_void, u64, u32, *mut *mut c_void, *mut u32) -> i32;
    pub type ClGetDeviceInfo =
        unsafe extern "system" fn(*mut c_void, u32, usize, *mut c_void, *mut usize) -> i32;
    pub type ClCreateContext = unsafe extern "system" fn(
        *const isize,
        u32,
        *const *mut c_void,
        Option<unsafe extern "system" fn(*const i8, *const c_void, usize, *mut c_void)>,
        *mut c_void,
        *mut i32,
    ) -> *mut c_void;
    pub type ClCreateCommandQueue =
        unsafe extern "system" fn(*mut c_void, *mut c_void, u64, *mut i32) -> *mut c_void;
    pub type ClCreateProgramWithSource = unsafe extern "system" fn(
        *mut c_void,
        u32,
        *const *const u8,
        *const usize,
        *mut i32,
    ) -> *mut c_void;
    pub type ClBuildProgram = unsafe extern "system" fn(
        *mut c_void,
        u32,
        *const *mut c_void,
        *const u8,
        Option<unsafe extern "system" fn(*mut c_void, *mut c_void)>,
        *mut c_void,
    ) -> i32;
    pub type ClCreateKernel =
        unsafe extern "system" fn(*mut c_void, *const u8, *mut i32) -> *mut c_void;
    pub type ClCreateBuffer =
        unsafe extern "system" fn(*mut c_void, u64, usize, *mut c_void, *mut i32) -> *mut c_void;
    pub type ClEnqueueWriteBuffer = unsafe extern "system" fn(
        *mut c_void,
        *mut c_void,
        u32,
        usize,
        usize,
        *const c_void,
        u32,
        *const *mut c_void,
        *mut *mut c_void,
    ) -> i32;
    pub type ClEnqueueReadBuffer = unsafe extern "system" fn(
        *mut c_void,
        *mut c_void,
        u32,
        usize,
        usize,
        *mut c_void,
        u32,
        *const *mut c_void,
        *mut *mut c_void,
    ) -> i32;
    pub type ClSetKernelArg =
        unsafe extern "system" fn(*mut c_void, u32, usize, *const c_void) -> i32;
    pub type ClEnqueueNDRangeKernel = unsafe extern "system" fn(
        *mut c_void,
        *mut c_void,
        u32,
        *const usize,
        *const usize,
        *const usize,
        u32,
        *const *mut c_void,
        *mut *mut c_void,
    ) -> i32;
    pub type ClFinish = unsafe extern "system" fn(*mut c_void) -> i32;
    pub type ClReleaseMemObject = unsafe extern "system" fn(*mut c_void) -> i32;

    pub struct OpenClApi {
        pub get_platform_ids: ClGetPlatformIDs,
        pub get_device_ids: ClGetDeviceIDs,
        pub get_device_info: ClGetDeviceInfo,
        pub create_context: ClCreateContext,
        pub create_command_queue: ClCreateCommandQueue,
        pub create_program_with_source: ClCreateProgramWithSource,
        pub build_program: ClBuildProgram,
        pub create_kernel: ClCreateKernel,
        pub create_buffer: ClCreateBuffer,
        pub enqueue_write_buffer: ClEnqueueWriteBuffer,
        pub enqueue_read_buffer: ClEnqueueReadBuffer,
        pub set_kernel_arg: ClSetKernelArg,
        pub enqueue_nd_range_kernel: ClEnqueueNDRangeKernel,
        pub finish: ClFinish,
        pub release_mem_object: ClReleaseMemObject,
    }

    impl OpenClApi {
        #[cfg(windows)]
        pub unsafe fn load() -> Option<Self> {
            unsafe {
                let lib = LoadLibraryA(PCSTR(b"OpenCL.dll\0".as_ptr())).ok()?;
                macro_rules! sym {
                    ($name:literal, $ty:ty) => {{
                        let addr = GetProcAddress(lib, PCSTR(concat!($name, "\0").as_ptr()))?;
                        std::mem::transmute::<_, $ty>(addr)
                    }};
                }
                Some(Self {
                    get_platform_ids: sym!("clGetPlatformIDs", ClGetPlatformIDs),
                    get_device_ids: sym!("clGetDeviceIDs", ClGetDeviceIDs),
                    get_device_info: sym!("clGetDeviceInfo", ClGetDeviceInfo),
                    create_context: sym!("clCreateContext", ClCreateContext),
                    create_command_queue: sym!("clCreateCommandQueue", ClCreateCommandQueue),
                    create_program_with_source: sym!(
                        "clCreateProgramWithSource",
                        ClCreateProgramWithSource
                    ),
                    build_program: sym!("clBuildProgram", ClBuildProgram),
                    create_kernel: sym!("clCreateKernel", ClCreateKernel),
                    create_buffer: sym!("clCreateBuffer", ClCreateBuffer),
                    enqueue_write_buffer: sym!("clEnqueueWriteBuffer", ClEnqueueWriteBuffer),
                    enqueue_read_buffer: sym!("clEnqueueReadBuffer", ClEnqueueReadBuffer),
                    set_kernel_arg: sym!("clSetKernelArg", ClSetKernelArg),
                    enqueue_nd_range_kernel: sym!(
                        "clEnqueueNDRangeKernel",
                        ClEnqueueNDRangeKernel
                    ),
                    finish: sym!("clFinish", ClFinish),
                    release_mem_object: sym!("clReleaseMemObject", ClReleaseMemObject),
                })
            }
        }

        #[cfg(unix)]
        pub unsafe fn load() -> Option<Self> {
            #[cfg_attr(any(target_os = "linux", target_os = "android"), link(name = "dl"))]
            unsafe extern "C" {
                fn dlopen(filename: *const u8, flags: i32) -> *mut c_void;
                fn dlsym(handle: *mut c_void, symbol: *const u8) -> *mut c_void;
            }
            const RTLD_LAZY: i32 = 1;
            const CANDIDATES: [&[u8]; 4] = [
                b"libOpenCL.so.1\0",
                b"libOpenCL.so\0",
                b"/System/Library/Frameworks/OpenCL.framework/OpenCL\0",
                b"/vendor/lib64/libOpenCL.so\0",
            ];
            unsafe {
                let mut lib: *mut c_void = core::ptr::null_mut();
                for cand in CANDIDATES {
                    lib = dlopen(cand.as_ptr(), RTLD_LAZY);
                    if !lib.is_null() {
                        break;
                    }
                }
                if lib.is_null() {
                    return None;
                }
                macro_rules! sym {
                    ($name:literal, $ty:ty) => {{
                        let addr = dlsym(lib, concat!($name, "\0").as_ptr());
                        if addr.is_null() {
                            return None;
                        }
                        core::mem::transmute::<*mut c_void, $ty>(addr)
                    }};
                }
                Some(Self {
                    get_platform_ids: sym!("clGetPlatformIDs", ClGetPlatformIDs),
                    get_device_ids: sym!("clGetDeviceIDs", ClGetDeviceIDs),
                    get_device_info: sym!("clGetDeviceInfo", ClGetDeviceInfo),
                    create_context: sym!("clCreateContext", ClCreateContext),
                    create_command_queue: sym!("clCreateCommandQueue", ClCreateCommandQueue),
                    create_program_with_source: sym!(
                        "clCreateProgramWithSource",
                        ClCreateProgramWithSource
                    ),
                    build_program: sym!("clBuildProgram", ClBuildProgram),
                    create_kernel: sym!("clCreateKernel", ClCreateKernel),
                    create_buffer: sym!("clCreateBuffer", ClCreateBuffer),
                    enqueue_write_buffer: sym!("clEnqueueWriteBuffer", ClEnqueueWriteBuffer),
                    enqueue_read_buffer: sym!("clEnqueueReadBuffer", ClEnqueueReadBuffer),
                    set_kernel_arg: sym!("clSetKernelArg", ClSetKernelArg),
                    enqueue_nd_range_kernel: sym!(
                        "clEnqueueNDRangeKernel",
                        ClEnqueueNDRangeKernel
                    ),
                    finish: sym!("clFinish", ClFinish),
                    release_mem_object: sym!("clReleaseMemObject", ClReleaseMemObject),
                })
            }
        }
    }
}

const NTT_OPENCL_KERNEL_SRC: &[u8] = b"
__kernel void ntt_bit_reverse(__global const uint* src, __global uint* dst, uint log_n) {
    uint i = (uint)get_global_id(0);
    uint rev = 0u;
    uint x = i;
    for (uint b = 0u; b < log_n; ++b) {
        rev = (rev << 1u) | (x & 1u);
        x >>= 1u;
    }
    dst[rev] = src[i];
}

__kernel void ntt_butterfly_stage(__global uint* data, uint half_len, uint w_len, uint mod_p) {
    uint tid = (uint)get_global_id(0);
    uint j = tid & (half_len - 1u);
    uint idx0 = ((tid - j) << 1u) + j;
    uint idx1 = idx0 + half_len;

    ulong w = 1ul;
    ulong base = (ulong)w_len;
    uint exp = j;
    while (exp > 0u) {
        if ((exp & 1u) != 0u) {
            w = (w * base) % (ulong)mod_p;
        }
        base = (base * base) % (ulong)mod_p;
        exp >>= 1u;
    }

    uint u = data[idx0];
    uint v = (uint)(((ulong)data[idx1] * w) % (ulong)mod_p);
    uint sum = u + v;
    if (sum >= mod_p) sum -= mod_p;
    uint diff = (u >= v) ? (u - v) : (u + mod_p - v);
    data[idx0] = sum;
    data[idx1] = diff;
}

__kernel void ntt_pointwise_mul_scale(__global uint* a, __global const uint* b, uint inv_n, uint mod_p) {
    uint i = (uint)get_global_id(0);
    ulong prod = ((ulong)a[i] * (ulong)b[i]) % (ulong)mod_p;
    a[i] = (uint)((prod * (ulong)inv_n) % (ulong)mod_p);
}

__kernel void ntt_pointwise_sqr_scale(__global uint* a, uint inv_n, uint mod_p) {
    uint i = (uint)get_global_id(0);
    ulong v = (ulong)a[i];
    ulong prod = (v * v) % (ulong)mod_p;
    a[i] = (uint)((prod * (ulong)inv_n) % (ulong)mod_p);
}
";

#[cfg(any(windows, unix))]
struct GpuSession {
    api: opencl_ffi::OpenClApi,
    context: *mut c_void,
    queue: *mut c_void,
    k_bit_reverse: *mut c_void,
    k_butterfly: *mut c_void,
    k_pointwise_mul: *mut c_void,
    k_pointwise_sqr: *mut c_void,
    buf_a: *mut c_void,
    buf_b: *mut c_void,
    buf_tmp: *mut c_void,
    buf_cap_elems: usize,
    device_name: String,
    compute_units: u32,
}

#[cfg(any(windows, unix))]
unsafe impl Send for GpuSession {}
#[cfg(any(windows, unix))]
unsafe impl Sync for GpuSession {}

#[cfg(any(windows, unix))]
impl GpuSession {
    unsafe fn ensure_buffers(&mut self, len: usize) -> bool {
        use opencl_ffi::*;
        if self.buf_cap_elems >= len && !self.buf_a.is_null() {
            return true;
        }
        unsafe {
            let _ = (self.api.finish)(self.queue);
            if !self.buf_a.is_null() {
                (self.api.release_mem_object)(self.buf_a);
                self.buf_a = std::ptr::null_mut();
            }
            if !self.buf_b.is_null() {
                (self.api.release_mem_object)(self.buf_b);
                self.buf_b = std::ptr::null_mut();
            }
            if !self.buf_tmp.is_null() {
                (self.api.release_mem_object)(self.buf_tmp);
                self.buf_tmp = std::ptr::null_mut();
            }

            let new_cap = len.next_power_of_two().max(65_536);
            let byte_size = new_cap * std::mem::size_of::<u32>();
            let mut err1 = 0i32;
            let mut err2 = 0i32;
            let mut err3 = 0i32;
            let ba = (self.api.create_buffer)(
                self.context,
                CL_MEM_READ_WRITE,
                byte_size,
                std::ptr::null_mut(),
                &mut err1,
            );
            let bb = (self.api.create_buffer)(
                self.context,
                CL_MEM_READ_WRITE,
                byte_size,
                std::ptr::null_mut(),
                &mut err2,
            );
            let bt = (self.api.create_buffer)(
                self.context,
                CL_MEM_READ_WRITE,
                byte_size,
                std::ptr::null_mut(),
                &mut err3,
            );
            if err1 != CL_SUCCESS
                || err2 != CL_SUCCESS
                || err3 != CL_SUCCESS
                || ba.is_null()
                || bb.is_null()
                || bt.is_null()
            {
                if !ba.is_null() {
                    (self.api.release_mem_object)(ba);
                }
                if !bb.is_null() {
                    (self.api.release_mem_object)(bb);
                }
                if !bt.is_null() {
                    (self.api.release_mem_object)(bt);
                }
                self.buf_cap_elems = 0;
                return false;
            }
            self.buf_a = ba;
            self.buf_b = bb;
            self.buf_tmp = bt;
            self.buf_cap_elems = new_cap;
            true
        }
    }

    unsafe fn run_ntt_inplace(
        &self,
        src_buf: *mut c_void,
        dst_buf: *mut c_void,
        n: usize,
        mod_p: u32,
        root: u32,
        invert: bool,
    ) -> bool {
        use opencl_ffi::*;
        let log_n = n.trailing_zeros();
        let global_n = n;
        let global_half = n / 2;

        unsafe {
            // 1. Bit-reverse copy from src_buf -> dst_buf
            if (self.api.set_kernel_arg)(
                self.k_bit_reverse,
                0,
                std::mem::size_of::<*mut c_void>(),
                &src_buf as *const _ as *const c_void,
            ) != CL_SUCCESS
            {
                return false;
            }
            if (self.api.set_kernel_arg)(
                self.k_bit_reverse,
                1,
                std::mem::size_of::<*mut c_void>(),
                &dst_buf as *const _ as *const c_void,
            ) != CL_SUCCESS
            {
                return false;
            }
            if (self.api.set_kernel_arg)(
                self.k_bit_reverse,
                2,
                std::mem::size_of::<u32>(),
                &log_n as *const _ as *const c_void,
            ) != CL_SUCCESS
            {
                return false;
            }
            if (self.api.enqueue_nd_range_kernel)(
                self.queue,
                self.k_bit_reverse,
                1,
                std::ptr::null(),
                &global_n,
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            ) != CL_SUCCESS
            {
                return false;
            }

            // 2. Butterfly stages in-place on dst_buf
            if (self.api.set_kernel_arg)(
                self.k_butterfly,
                0,
                std::mem::size_of::<*mut c_void>(),
                &dst_buf as *const _ as *const c_void,
            ) != CL_SUCCESS
            {
                return false;
            }
            if (self.api.set_kernel_arg)(
                self.k_butterfly,
                3,
                std::mem::size_of::<u32>(),
                &mod_p as *const _ as *const c_void,
            ) != CL_SUCCESS
            {
                return false;
            }

            let mut len = 2usize;
            while len <= n {
                if crate::engine::evaluator::is_task_aborted() {
                    return false;
                }
                let half_len = (len / 2) as u32;
                let mut w_len = pow_mod_u32(root, (mod_p - 1) / (len as u32), mod_p);
                if invert {
                    w_len = pow_mod_u32(w_len, mod_p - 2, mod_p);
                }
                if (self.api.set_kernel_arg)(
                    self.k_butterfly,
                    1,
                    std::mem::size_of::<u32>(),
                    &half_len as *const _ as *const c_void,
                ) != CL_SUCCESS
                {
                    return false;
                }
                if (self.api.set_kernel_arg)(
                    self.k_butterfly,
                    2,
                    std::mem::size_of::<u32>(),
                    &w_len as *const _ as *const c_void,
                ) != CL_SUCCESS
                {
                    return false;
                }
                if (self.api.enqueue_nd_range_kernel)(
                    self.queue,
                    self.k_butterfly,
                    1,
                    std::ptr::null(),
                    &global_half,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                ) != CL_SUCCESS
                {
                    return false;
                }
                len <<= 1;
            }
            true
        }
    }

    unsafe fn convolve_mod(
        &mut self,
        a: &[u32],
        b: Option<&[u32]>,
        out: &mut [u32],
        mod_p: u32,
        root: u32,
    ) -> bool {
        use opencl_ffi::*;
        if crate::engine::evaluator::is_task_aborted() {
            return false;
        }
        let n = out.len();
        if !n.is_power_of_two() || n < 2 || a.len() != n || b.is_some_and(|bs| bs.len() != n) {
            return false;
        }
        unsafe {
            if !self.ensure_buffers(n) {
                return false;
            }
            let byte_len = n * std::mem::size_of::<u32>();

            // Upload input A to buf_tmp
            if (self.api.enqueue_write_buffer)(
                self.queue,
                self.buf_tmp,
                CL_FALSE,
                0,
                byte_len,
                a.as_ptr() as *const c_void,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            ) != CL_SUCCESS
            {
                return false;
            }

            // Forward NTT(A): buf_tmp -> buf_a
            if !self.run_ntt_inplace(self.buf_tmp, self.buf_a, n, mod_p, root, false) {
                let _ = (self.api.finish)(self.queue);
                return false;
            }
            if crate::engine::evaluator::is_task_aborted() {
                let _ = (self.api.finish)(self.queue);
                return false;
            }

            let inv_n = pow_mod_u32(n as u32, mod_p - 2, mod_p);
            let global_n = n;

            if let Some(b_slice) = b {
                // Upload input B to buf_tmp
                if (self.api.enqueue_write_buffer)(
                    self.queue,
                    self.buf_tmp,
                    CL_FALSE,
                    0,
                    byte_len,
                    b_slice.as_ptr() as *const c_void,
                    0,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                ) != CL_SUCCESS
                {
                    let _ = (self.api.finish)(self.queue);
                    return false;
                }

                // Forward NTT(B): buf_tmp -> buf_b
                if !self.run_ntt_inplace(self.buf_tmp, self.buf_b, n, mod_p, root, false) {
                    let _ = (self.api.finish)(self.queue);
                    return false;
                }
                if crate::engine::evaluator::is_task_aborted() {
                    let _ = (self.api.finish)(self.queue);
                    return false;
                }

                // Pointwise multiply + scale by inv_n: buf_a = (buf_a * buf_b * inv_n) % mod_p
                if (self.api.set_kernel_arg)(
                    self.k_pointwise_mul,
                    0,
                    std::mem::size_of::<*mut c_void>(),
                    &self.buf_a as *const _ as *const c_void,
                ) != CL_SUCCESS
                    || (self.api.set_kernel_arg)(
                        self.k_pointwise_mul,
                        1,
                        std::mem::size_of::<*mut c_void>(),
                        &self.buf_b as *const _ as *const c_void,
                    ) != CL_SUCCESS
                    || (self.api.set_kernel_arg)(
                        self.k_pointwise_mul,
                        2,
                        std::mem::size_of::<u32>(),
                        &inv_n as *const _ as *const c_void,
                    ) != CL_SUCCESS
                    || (self.api.set_kernel_arg)(
                        self.k_pointwise_mul,
                        3,
                        std::mem::size_of::<u32>(),
                        &mod_p as *const _ as *const c_void,
                    ) != CL_SUCCESS
                {
                    let _ = (self.api.finish)(self.queue);
                    return false;
                }
                if (self.api.enqueue_nd_range_kernel)(
                    self.queue,
                    self.k_pointwise_mul,
                    1,
                    std::ptr::null(),
                    &global_n,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                ) != CL_SUCCESS
                {
                    let _ = (self.api.finish)(self.queue);
                    return false;
                }
            } else {
                // Squaring: buf_a = (buf_a * buf_a * inv_n) % mod_p
                if (self.api.set_kernel_arg)(
                    self.k_pointwise_sqr,
                    0,
                    std::mem::size_of::<*mut c_void>(),
                    &self.buf_a as *const _ as *const c_void,
                ) != CL_SUCCESS
                    || (self.api.set_kernel_arg)(
                        self.k_pointwise_sqr,
                        1,
                        std::mem::size_of::<u32>(),
                        &inv_n as *const _ as *const c_void,
                    ) != CL_SUCCESS
                    || (self.api.set_kernel_arg)(
                        self.k_pointwise_sqr,
                        2,
                        std::mem::size_of::<u32>(),
                        &mod_p as *const _ as *const c_void,
                    ) != CL_SUCCESS
                {
                    let _ = (self.api.finish)(self.queue);
                    return false;
                }
                if (self.api.enqueue_nd_range_kernel)(
                    self.queue,
                    self.k_pointwise_sqr,
                    1,
                    std::ptr::null(),
                    &global_n,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                ) != CL_SUCCESS
                {
                    let _ = (self.api.finish)(self.queue);
                    return false;
                }
            }

            // Inverse NTT: buf_a -> buf_tmp
            if !self.run_ntt_inplace(self.buf_a, self.buf_tmp, n, mod_p, root, true) {
                let _ = (self.api.finish)(self.queue);
                return false;
            }
            if crate::engine::evaluator::is_task_aborted() {
                let _ = (self.api.finish)(self.queue);
                return false;
            }

            // Read back result from buf_tmp into out
            if (self.api.enqueue_read_buffer)(
                self.queue,
                self.buf_tmp,
                CL_TRUE,
                0,
                byte_len,
                out.as_mut_ptr() as *mut c_void,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            ) != CL_SUCCESS
            {
                let _ = (self.api.finish)(self.queue);
                return false;
            }

            true
        }
    }
}

pub struct GpuComputeEngine {
    #[cfg(any(windows, unix))]
    session: Option<Mutex<GpuSession>>,
    pub available: bool,
    pub device_name: String,
    pub compute_units: u32,
}

static GPU_ENGINE: OnceLock<GpuComputeEngine> = OnceLock::new();

impl GpuComputeEngine {
    pub fn global() -> &'static Self {
        GPU_ENGINE.get_or_init(Self::init)
    }

    fn init() -> Self {
        #[cfg(any(windows, unix))]
        unsafe {
            use opencl_ffi::*;
            if let Some(api) = OpenClApi::load() {
                let mut num_platforms = 0u32;
                if (api.get_platform_ids)(0, std::ptr::null_mut(), &mut num_platforms) == CL_SUCCESS
                    && num_platforms > 0
                {
                    let mut platforms = vec![std::ptr::null_mut(); num_platforms as usize];
                    let _ = (api.get_platform_ids)(
                        num_platforms,
                        platforms.as_mut_ptr(),
                        &mut num_platforms,
                    );

                    // Collect all GPU devices across platforms and sort by discrete priority + compute units
                    let mut candidates: Vec<(u64, *mut c_void, *mut c_void, String, u32)> = Vec::new();

                    for &plat in &platforms {
                        if plat.is_null() {
                            continue;
                        }
                        let mut num_devs = 0u32;
                        if (api.get_device_ids)(
                            plat,
                            CL_DEVICE_TYPE_GPU,
                            0,
                            std::ptr::null_mut(),
                            &mut num_devs,
                        ) == CL_SUCCESS
                            && num_devs > 0
                        {
                            let mut devs = vec![std::ptr::null_mut(); num_devs as usize];
                            let _ = (api.get_device_ids)(
                                plat,
                                CL_DEVICE_TYPE_GPU,
                                num_devs,
                                devs.as_mut_ptr(),
                                &mut num_devs,
                            );
                            for &dev in &devs {
                                if dev.is_null() {
                                    continue;
                                }
                                let mut cu = 0u32;
                                let _ = (api.get_device_info)(
                                    dev,
                                    CL_DEVICE_MAX_COMPUTE_UNITS,
                                    std::mem::size_of::<u32>(),
                                    &mut cu as *mut _ as *mut c_void,
                                    std::ptr::null_mut(),
                                );
                                let mut vendor_id = 0u32;
                                let _ = (api.get_device_info)(
                                    dev,
                                    CL_DEVICE_VENDOR_ID,
                                    std::mem::size_of::<u32>(),
                                    &mut vendor_id as *mut _ as *mut c_void,
                                    std::ptr::null_mut(),
                                );
                                let mut name_buf = [0u8; 256];
                                let mut name_len = 0usize;
                                let _ = (api.get_device_info)(
                                    dev,
                                    CL_DEVICE_NAME,
                                    name_buf.len(),
                                    name_buf.as_mut_ptr() as *mut c_void,
                                    &mut name_len,
                                );
                                let name = String::from_utf8_lossy(
                                    &name_buf[..name_len.min(name_buf.len())],
                                )
                                .trim_matches(char::from(0))
                                .trim()
                                .to_string();

                                // Discrete NVIDIA (0x10DE) or AMD (0x1002) gets priority
                                let is_discrete = vendor_id == 0x10DE || vendor_id == 0x1002;
                                let score =
                                    (if is_discrete { 10_000u64 } else { 100u64 }) + (cu as u64);
                                candidates.push((score, plat, dev, name, cu));
                            }
                        }
                    }

                    candidates.sort_by(|a, b| b.0.cmp(&a.0));

                    for (_score, best_plat, best_dev, best_name, best_cu) in candidates {
                        const CL_CONTEXT_PLATFORM: isize = 0x1084;
                        let ctx_props: [isize; 3] = [CL_CONTEXT_PLATFORM, best_plat as isize, 0];
                        let mut err = 0i32;
                        let mut ctx = (api.create_context)(
                            ctx_props.as_ptr(),
                            1,
                            &best_dev,
                            None,
                            std::ptr::null_mut(),
                            &mut err,
                        );
                        if err != CL_SUCCESS || ctx.is_null() {
                            ctx = (api.create_context)(
                                std::ptr::null(),
                                1,
                                &best_dev,
                                None,
                                std::ptr::null_mut(),
                                &mut err,
                            );
                        }
                        if err == CL_SUCCESS && !ctx.is_null() {
                            let queue = (api.create_command_queue)(ctx, best_dev, 0, &mut err);
                            if err == CL_SUCCESS && !queue.is_null() {
                                let src_ptr = NTT_OPENCL_KERNEL_SRC.as_ptr();
                                let src_len = NTT_OPENCL_KERNEL_SRC.len();
                                let prog = (api.create_program_with_source)(
                                    ctx,
                                    1,
                                    &src_ptr,
                                    &src_len,
                                    &mut err,
                                );
                                if err == CL_SUCCESS && !prog.is_null() {
                                    let build_res = (api.build_program)(
                                        prog,
                                        1,
                                        &best_dev,
                                        std::ptr::null(),
                                        None,
                                        std::ptr::null_mut(),
                                    );
                                    if build_res == CL_SUCCESS {
                                        let mut e1 = 0;
                                        let mut e2 = 0;
                                        let mut e3 = 0;
                                        let mut e4 = 0;
                                        let k_rev = (api.create_kernel)(
                                            prog,
                                            b"ntt_bit_reverse\0".as_ptr(),
                                            &mut e1,
                                        );
                                        let k_bfly = (api.create_kernel)(
                                            prog,
                                            b"ntt_butterfly_stage\0".as_ptr(),
                                            &mut e2,
                                        );
                                        let k_mul = (api.create_kernel)(
                                            prog,
                                            b"ntt_pointwise_mul_scale\0".as_ptr(),
                                            &mut e3,
                                        );
                                        let k_sqr = (api.create_kernel)(
                                            prog,
                                            b"ntt_pointwise_sqr_scale\0".as_ptr(),
                                            &mut e4,
                                        );
                                        if e1 == CL_SUCCESS
                                            && e2 == CL_SUCCESS
                                            && e3 == CL_SUCCESS
                                            && e4 == CL_SUCCESS
                                            && !k_rev.is_null()
                                            && !k_bfly.is_null()
                                            && !k_mul.is_null()
                                            && !k_sqr.is_null()
                                        {
                                            let api_clone = OpenClApi::load().unwrap();
                                            let mut sess = GpuSession {
                                                api: api_clone,
                                                context: ctx,
                                                queue,
                                                k_bit_reverse: k_rev,
                                                k_butterfly: k_bfly,
                                                k_pointwise_mul: k_mul,
                                                k_pointwise_sqr: k_sqr,
                                                buf_a: std::ptr::null_mut(),
                                                buf_b: std::ptr::null_mut(),
                                                buf_tmp: std::ptr::null_mut(),
                                                buf_cap_elems: 0,
                                                device_name: best_name.clone(),
                                                compute_units: best_cu,
                                            };
                                            // Hardware self-verification on a small 8-point convolution
                                            // Guard with epoch 0 so a cancelled thread-local context on the first caller never poisons OnceLock init.
                                            let _init_guard = crate::engine::evaluator::EpochGuard::new(0);
                                            let a_test = [3u32, 5, 7, 11, 0, 0, 0, 0];
                                            let b_test = [2u32, 4, 6, 8, 0, 0, 0, 0];
                                            let mut out_test = [0u32; 8];
                                            let ok = sess.convolve_mod(
                                                &a_test,
                                                Some(&b_test),
                                                &mut out_test,
                                                998_244_353,
                                                3,
                                            );
                                            // Expected: (3 + 5x + 7x^2 + 11x^3)(2 + 4x + 6x^2 + 8x^3)
                                            // x^0: 3*2 = 6
                                            // x^1: 3*4 + 5*2 = 22
                                            // x^2: 3*6 + 5*4 + 7*2 = 52
                                            // x^3: 3*8 + 5*6 + 7*4 + 11*2 = 104
                                            // x^4: 5*8 + 7*6 + 11*4 = 40 + 42 + 44 = 126
                                            // x^5: 7*8 + 11*6 = 56 + 66 = 122
                                            // x^6: 11*8 = 88
                                            if ok
                                                && out_test
                                                    == [6, 22, 52, 104, 126, 122, 88, 0]
                                            {
                                                return Self {
                                                    session: Some(Mutex::new(sess)),
                                                    available: true,
                                                    device_name: best_name,
                                                    compute_units: best_cu,
                                                };
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Self {
            #[cfg(any(windows, unix))]
            session: None,
            available: false,
            device_name: String::new(),
            compute_units: 0,
        }
    }

    pub fn convolve_mod(
        &self,
        a: &[u32],
        b: Option<&[u32]>,
        out: &mut [u32],
        mod_p: u32,
        root: u32,
    ) -> bool {
        if !self.available || crate::engine::evaluator::is_task_aborted() {
            return false;
        }
        #[cfg(any(windows, unix))]
        if let Some(ref mtx) = self.session {
            let mut guard = mtx.lock().unwrap_or_else(|e| e.into_inner());
            if crate::engine::evaluator::is_task_aborted() {
                return false;
            }
            return unsafe { guard.convolve_mod(a, b, out, mod_p, root) };
        }
        false
    }

    #[inline]
    pub fn try_convolve_mod(
        &self,
        a: &[u32],
        b: Option<&[u32]>,
        out: &mut [u32],
        mod_p: u32,
        root: u32,
    ) -> bool {
        self.convolve_mod(a, b, out, mod_p, root)
    }
}
