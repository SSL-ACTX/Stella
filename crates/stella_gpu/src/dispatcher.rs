// crates/stella_gpu/src/dispatcher.rs
//! Dynamic Compute Backend Discovery and Dispatcher.

use crate::backends::*;
use crate::traits::GpuComputeBackend;
use std::sync::Arc;

/// Automatically detects and selects the fastest operational compute backend.
///
/// Priority order:
/// 1. Android Wireless ADB Mali GPU Bridge (if running on Android and adbd port is active)
/// 2. Native Desktop/Server Vulkan (if native-vulkan feature is enabled and GPU is present)
/// 3. Universal Multi-threaded CPU SIMD Engine (Rayon + NEON/AVX)
pub fn auto_detect_backend() -> Arc<dyn GpuComputeBackend> {
    #[cfg(all(target_os = "android", feature = "android-radb"))]
    {
        if let Ok(backend) = AndroidRadbBackend::try_connect() {
            if backend.is_available() {
                return Arc::new(backend);
            }
        }
    }

    #[cfg(feature = "native-vulkan")]
    {
        let native = NativeVulkanBackend::new();
        if native.is_available() {
            return Arc::new(native);
        }
    }

    Arc::new(CpuSimdBackend::new())
}

/// Explicitly instantiate the Android radb hardware GPU backend if available.
#[cfg(all(target_os = "android", feature = "android-radb"))]
pub fn get_android_gpu_backend() -> Result<AndroidRadbBackend, String> {
    AndroidRadbBackend::try_connect()
}
