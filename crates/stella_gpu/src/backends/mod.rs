// crates/stella_gpu/src/backends/mod.rs
pub mod cpu_simd;

#[cfg(all(target_os = "android", feature = "android-radb"))]
pub mod android_radb;

#[cfg(feature = "native-vulkan")]
pub mod native_vulkan;

#[cfg(all(target_os = "android", feature = "android-radb"))]
pub use android_radb::AndroidRadbBackend;
pub use cpu_simd::CpuSimdBackend;
#[cfg(feature = "native-vulkan")]
pub use native_vulkan::NativeVulkanBackend;
