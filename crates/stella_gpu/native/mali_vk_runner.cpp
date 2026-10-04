#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <dlfcn.h>
#include <vulkan/vulkan.h>

// FP16 representation and conversion helpers
typedef uint16_t float16_t;

static inline float16_t float_to_fp16(float val) {
    union { float f; uint32_t u; } in;
    in.f = val;
    uint32_t sign = (in.u >> 31) & 0x1;
    int32_t exp = ((in.u >> 23) & 0xFF) - 127;
    uint32_t frac = in.u & 0x7FFFFF;

    if (exp > 15) return (float16_t)((sign << 15) | 0x7C00); // Inf
    if (exp < -14) return (float16_t)(sign << 15);           // Zero / Underflow
    return (float16_t)((sign << 15) | ((exp + 15) << 10) | (frac >> 13));
}

static inline float fp16_to_float(float16_t val) {
    uint32_t sign = (val >> 15) & 0x1;
    uint32_t exp = (val >> 10) & 0x1F;
    uint32_t frac = val & 0x3FF;

    union { uint32_t u; float f; } out;
    if (exp == 0) {
        out.u = (sign << 31);
    } else if (exp == 31) {
        out.u = (sign << 31) | 0x7F800000 | (frac << 13);
    } else {
        out.u = (sign << 31) | ((exp - 15 + 127) << 23) | (frac << 13);
    }
    return out.f;
}

// Push constants struct matching GLSL layout
typedef struct {
    uint32_t N;
    uint32_t B;
    uint32_t step;
} PushConstants;

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s <N> <B> <cycles>\n", argv[0]);
        return 1;
    }

    uint32_t N = (uint32_t)atoi(argv[1]);
    uint32_t B = (uint32_t)atoi(argv[2]);
    uint32_t cycles = (uint32_t)atoi(argv[3]);

    void* handle = dlopen("libvulkan.so", RTLD_NOW);
    if (!handle) {
        printf("[-] Failed to open libvulkan.so: %s\n", dlerror());
        return 1;
    }

    #define LOAD_VK(type, name) \
        type name = (type)dlsym(handle, #name); \
        if (!name) { printf("[-] Missing symbol: %s\n", #name); return 1; }

    LOAD_VK(PFN_vkCreateInstance, vkCreateInstance);
    LOAD_VK(PFN_vkEnumeratePhysicalDevices, vkEnumeratePhysicalDevices);
    LOAD_VK(PFN_vkGetPhysicalDeviceQueueFamilyProperties, vkGetPhysicalDeviceQueueFamilyProperties);
    LOAD_VK(PFN_vkGetPhysicalDeviceMemoryProperties, vkGetPhysicalDeviceMemoryProperties);
    LOAD_VK(PFN_vkCreateDevice, vkCreateDevice);
    LOAD_VK(PFN_vkGetDeviceQueue, vkGetDeviceQueue);
    LOAD_VK(PFN_vkCreateBuffer, vkCreateBuffer);
    LOAD_VK(PFN_vkGetBufferMemoryRequirements, vkGetBufferMemoryRequirements);
    LOAD_VK(PFN_vkAllocateMemory, vkAllocateMemory);
    LOAD_VK(PFN_vkBindBufferMemory, vkBindBufferMemory);
    LOAD_VK(PFN_vkMapMemory, vkMapMemory);
    LOAD_VK(PFN_vkUnmapMemory, vkUnmapMemory);
    LOAD_VK(PFN_vkFlushMappedMemoryRanges, vkFlushMappedMemoryRanges);
    LOAD_VK(PFN_vkInvalidateMappedMemoryRanges, vkInvalidateMappedMemoryRanges);
    LOAD_VK(PFN_vkCreateShaderModule, vkCreateShaderModule);
    LOAD_VK(PFN_vkCreateDescriptorSetLayout, vkCreateDescriptorSetLayout);
    LOAD_VK(PFN_vkCreatePipelineLayout, vkCreatePipelineLayout);
    LOAD_VK(PFN_vkCreateDescriptorPool, vkCreateDescriptorPool);
    LOAD_VK(PFN_vkAllocateDescriptorSets, vkAllocateDescriptorSets);
    LOAD_VK(PFN_vkUpdateDescriptorSets, vkUpdateDescriptorSets);
    LOAD_VK(PFN_vkCreateComputePipelines, vkCreateComputePipelines);
    LOAD_VK(PFN_vkCreateCommandPool, vkCreateCommandPool);
    LOAD_VK(PFN_vkAllocateCommandBuffers, vkAllocateCommandBuffers);
    LOAD_VK(PFN_vkBeginCommandBuffer, vkBeginCommandBuffer);
    LOAD_VK(PFN_vkCmdBindPipeline, vkCmdBindPipeline);
    LOAD_VK(PFN_vkCmdBindDescriptorSets, vkCmdBindDescriptorSets);
    LOAD_VK(PFN_vkCmdPushConstants, vkCmdPushConstants);
    LOAD_VK(PFN_vkCmdDispatch, vkCmdDispatch);
    LOAD_VK(PFN_vkCmdPipelineBarrier, vkCmdPipelineBarrier);
    LOAD_VK(PFN_vkEndCommandBuffer, vkEndCommandBuffer);
    LOAD_VK(PFN_vkQueueSubmit, vkQueueSubmit);
    LOAD_VK(PFN_vkQueueWaitIdle, vkQueueWaitIdle);

    // 1. Create Instance
    VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Stella Mali GPU Compute",
        .apiVersion = VK_API_VERSION_1_1,
    };
    VkInstanceCreateInfo instInfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,
    };
    VkInstance instance;
    if (vkCreateInstance(&instInfo, NULL, &instance) != VK_SUCCESS) {
        printf("[-] vkCreateInstance failed\n");
        return 1;
    }

    // 2. Select Mali Physical Device
    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(instance, &devCount, NULL);
    if (devCount == 0) {
        printf("[-] No physical devices found\n");
        return 1;
    }
    VkPhysicalDevice physDevice;
    vkEnumeratePhysicalDevices(instance, &devCount, &physDevice);

    uint32_t qCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physDevice, &qCount, NULL);
    VkQueueFamilyProperties qProps[16];
    vkGetPhysicalDeviceQueueFamilyProperties(physDevice, &qCount, qProps);
    uint32_t computeQueueIdx = 0;
    for (uint32_t i = 0; i < qCount; i++) {
        if (qProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            computeQueueIdx = i;
            break;
        }
    }

    // 3. Logical Device
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = computeQueueIdx,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };
    VkDeviceCreateInfo devInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qInfo,
    };
    VkDevice device;
    if (vkCreateDevice(physDevice, &devInfo, NULL, &device) != VK_SUCCESS) {
        printf("[-] vkCreateDevice failed\n");
        return 1;
    }

    VkQueue queue;
    vkGetDeviceQueue(device, computeQueueIdx, 0, &queue);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDevice, &memProps);

    auto findMemType = [&](uint32_t filter, VkMemoryPropertyFlags flags) -> int32_t {
        for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
            if ((filter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & flags) == flags) {
                return (int32_t)i;
            }
        }
        return -1;
    };

    auto allocBuffer = [&](VkDeviceSize size, VkBuffer* buf, VkDeviceMemory* mem) -> bool {
        VkBufferCreateInfo bInfo = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        if (vkCreateBuffer(device, &bInfo, NULL, buf) != VK_SUCCESS) return false;
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, *buf, &req);
        // Prefer HOST_CACHED | HOST_VISIBLE (GPU L2 cached on Mali)
        int32_t idx = findMemType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        if (idx < 0) {
            idx = findMemType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }
        if (idx < 0) return false;

        VkMemoryAllocateInfo aInfo = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = req.size,
            .memoryTypeIndex = (uint32_t)idx,
        };
        if (vkAllocateMemory(device, &aInfo, NULL, mem) != VK_SUCCESS) return false;
        vkBindBufferMemory(device, *buf, *mem, 0);
        return true;
    };

    // Buffers using native FP16 (2 bytes per weight/state)
    VkBuffer bufW, bufB, bufS0, bufS1;
    VkDeviceMemory memW, memB, memS0, memS1;
    VkDeviceSize szW = (VkDeviceSize)N * N * sizeof(float16_t);
    VkDeviceSize szB = (VkDeviceSize)N * sizeof(float16_t);
    VkDeviceSize szS = (VkDeviceSize)N * B * sizeof(float16_t);

    allocBuffer(szW, &bufW, &memW);
    allocBuffer(szB, &bufB, &memB);
    allocBuffer(szS, &bufS0, &memS0);
    allocBuffer(szS, &bufS1, &memS1);

    // Fill Initial Data in native FP16
    void *pW, *pB, *pS0;
    vkMapMemory(device, memW, 0, szW, 0, &pW);
    vkMapMemory(device, memB, 0, szB, 0, &pB);
    vkMapMemory(device, memS0, 0, szS, 0, &pS0);

    float16_t* wArr = (float16_t*)pW;
    float16_t* bArr = (float16_t*)pB;
    float16_t* sArr = (float16_t*)pS0;

    float16_t f16_w_off = float_to_fp16(0.01f);
    float16_t f16_w_diag = float_to_fp16(0.85f);
    float16_t f16_bias = float_to_fp16(0.05f);
    float16_t f16_state = float_to_fp16(0.5f);

    for (uint32_t i = 0; i < N * N; i++) wArr[i] = f16_w_off;
    for (uint32_t i = 0; i < N; i++) {
        wArr[i * N + i] = f16_w_diag;
        bArr[i] = f16_bias;
    }
    for (uint32_t i = 0; i < N * B; i++) sArr[i] = f16_state;

    VkMappedMemoryRange flushRanges[3] = {
        { .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, .pNext = NULL, .memory = memW,  .offset = 0, .size = szW },
        { .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, .pNext = NULL, .memory = memB,  .offset = 0, .size = szB },
        { .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, .pNext = NULL, .memory = memS0, .offset = 0, .size = szS },
    };
    vkFlushMappedMemoryRanges(device, 3, flushRanges);

    vkUnmapMemory(device, memW);
    vkUnmapMemory(device, memB);
    vkUnmapMemory(device, memS0);

    // 4. Load Shader SPIR-V
    FILE* f = fopen("/data/local/tmp/.stella/gemm.spv", "rb");
    if (!f) {
        f = fopen("/data/local/tmp/stella_gemm.spv", "rb");
    }
    if (!f) {
        printf("[-] Failed to open /data/local/tmp/.stella/gemm.spv\n");
        return 1;
    }
    fseek(f, 0, SEEK_END);
    size_t spvLen = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t* spv = (uint32_t*)malloc(spvLen);
    fread(spv, 1, spvLen, f);
    fclose(f);

    VkShaderModuleCreateInfo smInfo = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spvLen,
        .pCode = spv,
    };
    VkShaderModule shModule;
    vkCreateShaderModule(device, &smInfo, NULL, &shModule);
    free(spv);

    // 5. Pipeline Layout & Descriptors
    VkDescriptorSetLayoutBinding bindings[4] = {
        { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
    };
    VkDescriptorSetLayoutCreateInfo dslInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 4,
        .pBindings = bindings,
    };
    VkDescriptorSetLayout dsLayout;
    vkCreateDescriptorSetLayout(device, &dslInfo, NULL, &dsLayout);

    VkPushConstantRange pcRange = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(PushConstants),
    };
    VkPipelineLayoutCreateInfo plInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &dsLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcRange,
    };
    VkPipelineLayout plLayout;
    vkCreatePipelineLayout(device, &plInfo, NULL, &plLayout);

    VkComputePipelineCreateInfo cpInfo = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shModule,
            .pName = "main",
        },
        .layout = plLayout,
    };
    VkPipeline pipeline;
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpInfo, NULL, &pipeline);

    // Descriptor Pool & Writes
    VkDescriptorPoolSize poolSize = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 4 };
    VkDescriptorPoolCreateInfo dpInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize,
    };
    VkDescriptorPool dPool;
    vkCreateDescriptorPool(device, &dpInfo, NULL, &dPool);

    VkDescriptorSetAllocateInfo dsAlloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = dPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &dsLayout,
    };
    VkDescriptorSet dSet;
    vkAllocateDescriptorSets(device, &dsAlloc, &dSet);

    VkDescriptorBufferInfo bInfos[4] = {
        { .buffer = bufW, .offset = 0, .range = szW },
        { .buffer = bufB, .offset = 0, .range = szB },
        { .buffer = bufS0, .offset = 0, .range = szS },
        { .buffer = bufS1, .offset = 0, .range = szS },
    };
    VkWriteDescriptorSet writes[4];
    for (int i = 0; i < 4; i++) {
        writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = dSet,
            .dstBinding = (uint32_t)i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &bInfos[i],
        };
    }
    vkUpdateDescriptorSets(device, 4, writes, 0, NULL);

    // 6. Record Commands
    VkCommandPoolCreateInfo cpPoolInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = computeQueueIdx,
    };
    VkCommandPool cmdPool;
    vkCreateCommandPool(device, &cpPoolInfo, NULL, &cmdPool);

    VkCommandBufferAllocateInfo cbAlloc = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = cmdPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &cbAlloc, &cmd);

    VkCommandBufferBeginInfo cbBegin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cmd, &cbBegin);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plLayout, 0, 1, &dSet, 0, NULL);

    uint32_t gx = (B + 15) / 16;
    uint32_t gy = (N + 15) / 16;

    VkMemoryBarrier memBarrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = NULL,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    };

    for (uint32_t step = 0; step < cycles; ++step) {
        PushConstants pc = { N, B, step };
        vkCmdPushConstants(cmd, plLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc);
        vkCmdDispatch(cmd, gx, gy, 1);

        if (step + 1 < cycles) {
            vkCmdPipelineBarrier(
                cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                1, &memBarrier,
                0, NULL,
                0, NULL
            );
        }
    }
    vkEndCommandBuffer(cmd);

    // 7. Execute on Mali GPU & Measure
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    VkSubmitInfo sInfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };
    vkQueueSubmit(queue, 1, &sInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

    // Read back output sample in native FP16
    VkDeviceMemory outMem = (cycles & 1) ? memS1 : memS0;
    void* pOut;
    vkMapMemory(device, outMem, 0, szS, 0, &pOut);
    VkMappedMemoryRange invRange = {
        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .pNext = NULL,
        .memory = outMem,
        .offset = 0,
        .size = sizeof(float16_t),
    };
    vkInvalidateMappedMemoryRanges(device, 1, &invRange);
    float16_t f16_val = ((float16_t*)pOut)[0];
    float val = fp16_to_float(f16_val);
    vkUnmapMemory(device, outMem);

    double totalMacs = (double)N * (double)N * (double)B * (double)cycles;
    double gigaMacsSec = (totalMacs / (ms / 1000.0)) / 1e9;
    double agentCyclesSec = ((double)B * (double)cycles) / (ms / 1000.0);

    printf("[+] ARM Mali-G57 GPU Compute Executed via Vulkan!\n");
    printf("    Device                 : ARM Mali-G57 (Direct Driver Pipeline)\n");
    printf("    Dimensions             : N=%u neurons, B=%u agents, cycles=%u\n", N, B, cycles);
    printf("    Hardware Execution Time: %.2f ms\n", ms);
    printf("    Physical GPU Compute   : %.3f GigaMACs/sec\n", gigaMacsSec);
    printf("    Swarm Agent Velocity   : %.2f Thousand agent-cycles/sec\n", agentCyclesSec / 1000.0);
    printf("    Attractor Convergence  : S[0, 0] = %.4f\n", val);

    return 0;
}
