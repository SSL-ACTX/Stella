#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

typedef int VkResult;
typedef void* VkInstance;
typedef void* VkPhysicalDevice;

typedef struct VkApplicationInfo {
    int sType;
    const void* pNext;
    const char* pApplicationName;
    unsigned int applicationVersion;
    const char* pEngineName;
    unsigned int engineVersion;
    unsigned int apiVersion;
} VkApplicationInfo;

typedef struct VkInstanceCreateInfo {
    int sType;
    const void* pNext;
    unsigned int flags;
    const VkApplicationInfo* pApplicationInfo;
    unsigned int enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    unsigned int enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
} VkInstanceCreateInfo;

typedef struct VkPhysicalDeviceProperties {
    unsigned int apiVersion;
    unsigned int driverVersion;
    unsigned int vendorID;
    unsigned int deviceID;
    int deviceType;
    char deviceName[256];
    unsigned char pipelineCacheUUID[16];
    char limits[512];
    char sparseProperties[20];
} VkPhysicalDeviceProperties;

int main() {
    void* handle = dlopen("libvulkan.so", RTLD_NOW);
    if (!handle) {
        printf("Failed to open libvulkan.so: %s\n", dlerror());
        return 1;
    }

    VkResult (*vkCreateInstance)(const VkInstanceCreateInfo*, const void*, VkInstance*) = 
        dlsym(handle, "vkCreateInstance");
    VkResult (*vkEnumeratePhysicalDevices)(VkInstance, unsigned int*, VkPhysicalDevice*) = 
        dlsym(handle, "vkEnumeratePhysicalDevices");
    void (*vkGetPhysicalDeviceProperties)(VkPhysicalDevice, VkPhysicalDeviceProperties*) = 
        dlsym(handle, "vkGetPhysicalDeviceProperties");
    void (*vkDestroyInstance)(VkInstance, const void*) = 
        dlsym(handle, "vkDestroyInstance");

    if (!vkCreateInstance || !vkEnumeratePhysicalDevices || !vkGetPhysicalDeviceProperties) {
        printf("Failed to locate Vulkan symbols in libvulkan.so\n");
        return 1;
    }

    VkApplicationInfo appInfo = {0};
    appInfo.sType = 1; // VK_STRUCTURE_TYPE_APPLICATION_INFO
    appInfo.pApplicationName = "Stella GPU Probe";
    appInfo.apiVersion = (1 << 22) | (1 << 12); // Vulkan 1.1

    VkInstanceCreateInfo createInfo = {0};
    createInfo.sType = 10; // VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
    createInfo.pApplicationInfo = &appInfo;

    VkInstance instance = NULL;
    VkResult res = vkCreateInstance(&createInfo, NULL, &instance);
    if (res != 0) {
        printf("vkCreateInstance failed with code %d\n", res);
        return 1;
    }

    unsigned int deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
    printf("Physical devices count in ADB shell: %u\n", deviceCount);

    typedef struct VkMemoryType {
        unsigned int propertyFlags;
        unsigned int heapIndex;
    } VkMemoryType;
    typedef struct VkMemoryHeap {
        unsigned long long size;
        unsigned int flags;
    } VkMemoryHeap;
    typedef struct VkPhysicalDeviceMemoryProperties {
        unsigned int memoryTypeCount;
        VkMemoryType memoryTypes[32];
        unsigned int memoryHeapCount;
        VkMemoryHeap memoryHeaps[16];
    } VkPhysicalDeviceMemoryProperties;

    void (*vkGetPhysicalDeviceMemoryProperties)(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties*) =
        dlsym(handle, "vkGetPhysicalDeviceMemoryProperties");

    if (deviceCount > 0) {
        VkPhysicalDevice* devices = malloc(sizeof(VkPhysicalDevice) * deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices);
        for (unsigned int i = 0; i < deviceCount; i++) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(devices[i], &props);
            printf("Hardware GPU [%u]: %s (Vendor: 0x%04x, Type: %d)\n",
                   i, props.deviceName, props.vendorID, props.deviceType);

            if (vkGetPhysicalDeviceMemoryProperties) {
                VkPhysicalDeviceMemoryProperties memProps;
                vkGetPhysicalDeviceMemoryProperties(devices[i], &memProps);
                printf("  Memory Types (%u total):\n", memProps.memoryTypeCount);
                for (unsigned int m = 0; m < memProps.memoryTypeCount; m++) {
                    unsigned int f = memProps.memoryTypes[m].propertyFlags;
                    printf("    [%u] heap %u, flags 0x%x (dev_local=%d, host_vis=%d, host_coh=%d, host_cached=%d)\n",
                           m, memProps.memoryTypes[m].heapIndex, f,
                           (f & 0x1) != 0,  // VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                           (f & 0x2) != 0,  // VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                           (f & 0x4) != 0,  // VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                           (f & 0x8) != 0); // VK_MEMORY_PROPERTY_HOST_CACHED_BIT
                }
            }
        }
        free(devices);
    }

    vkDestroyInstance(instance, NULL);
    dlclose(handle);
    return 0;
}
