#pragma once

#include <vulkan/vulkan.h> // Vulkan
#include <optional> // optional
#include <string.h> // strcmp
#include <assert.h> // assert
#include <cmath> // ceil

#ifdef NDEBUG
const auto enableValidationLayers = std::nullopt;
#else
const auto enableValidationLayers = std::optional<char const*>{"VK_LAYER_KHRONOS_validation"};
#endif

// Macro to check Vulkan result
#define VK_CHECK_RESULT(f) 																\
{																						\
    VkResult res = (f);																	\
    if (res != VK_SUCCESS)																\
    {																					\
        printf("Fatal : VkResult is %d in %s at line %d\n", res,  __FILE__, __LINE__);  \
        assert(res == VK_SUCCESS);														\
    }																					\
}

class ComputeApp {
    // -------------------------------------------------
    // Private members
    // -------------------------------------------------
    public:
        VkInstance instance;                        // Vulkan instance.
        VkPhysicalDevice physicalDevice;            // Physical device (e.g. GPU).
        VkDevice device;                            // Logical device by which we connect to our physical device.
        uint32_t queueFamilyIndex;                  // Index to a queue family.
        VkQueue queue;                              // Queue.
        VkBuffer buffer;                            // Buffer.
        VkDeviceMemory bufferMemory;                // Buffer's memory.
        VkDescriptorSetLayout descriptorSetLayout;  // Layout of a descriptor set.
        VkDescriptorPool descriptorPool;            // Pool from which to pull descriptor sets.
        VkDescriptorSet descriptorSet;              // Descriptor set.
        VkShaderModule computeShaderModule;         // Shader.
        VkPipelineLayout pipelineLayout;            // Layout for a pipeline.
        VkPipeline pipeline;                        // Pipeline.
        VkCommandPool commandPool;                  // Pool from which to pull command buffer.
        VkCommandBuffer commandBuffer;              // Command buffer.
    // -------------------------------------------------
    // Public methods
    // -------------------------------------------------
    public:
        ComputeApp(
            char const* shaderFile,
            uint32_t const bufferSize,
            float* bufferData,
            uint32_t* dims, // [x,y,z],
            uint32_t const* dimLengths // [local_size_x, local_size_y, local_size_z]
        );
        ~ComputeApp();
};

namespace Utility {
    // Creates Vulkan instance
    void createInstance(VkInstance& instance);
    // Gets physical device
    void getPhysicalDevice(VkInstance const& instance, VkPhysicalDevice& physicalDevice);
    // Gets an index to a queue family
     uint32_t getComputeQueueFamilyIndex(VkPhysicalDevice const& physicalDevice);
    // Creates logical device
    void createDevice(
        VkPhysicalDevice const& physicalDevice,
        uint32_t& queueFamilyIndex,
        VkDevice& device,
        VkQueue& queue
    );
    // Finds the memory type by which we can access memory allocated from the heap
     uint32_t findMemoryType(
        VkPhysicalDevice const& physicalDevice,
        uint32_t const memoryTypeBits,
        VkMemoryPropertyFlags const properties
    );
    // Creates buffer
    void createBuffer(
        VkPhysicalDevice const& physicalDevice,
        VkDevice const& device,
        uint32_t const size,
        VkBuffer * const buffer,
        VkDeviceMemory * const bufferMemory
    );
    // Fills a buffer with given data
    void fillBuffer(
        VkDevice const& device,
        VkDeviceMemory& bufferMemory,
        float*& bufferData, 
        uint32_t const bufferSize
    );
    // Creates descriptor set layout
    void createDescriptorSetLayout(
        VkDevice const& device, 
        VkDescriptorSetLayout* descriptorSetLayout
    );
    // Creates descriptor set
    void createDescriptorSet(
        VkDevice const& device,
        VkDescriptorPool* descriptorPool,
        VkDescriptorSetLayout* descriptorSetLayout,
        VkBuffer& buffer,
        VkDescriptorSet& descriptorSet
    );
    // Reads shader file
    std::pair<uint32_t,uint32_t*> readShader(char const* filename);
    // Creates compute pipeline
    void createComputePipeline(
        VkDevice const& device,
        char const* shaderFile,
        VkShaderModule* computeShaderModule,
        VkDescriptorSetLayout* descriptorSetLayout,
        VkPipelineLayout* pipelineLayout,
        VkPipeline* pipeline
    );
    // Creates command buffer
    void createCommandBuffer(
        uint32_t queueFamilyIndex,
        VkDevice& device,
        VkCommandPool* commandPool,
        VkCommandBuffer* commandBuffer,
        VkPipeline& pipeline,
        VkPipelineLayout& pipelineLayout,
        VkDescriptorSet& descriptorSet,
        uint32_t const* dims, // [x,y,z],
        uint32_t const* dimLengths // [local_size_x, local_size_y, local_size_z]
    );
    // Runs command buffer
    void runCommandBuffer(
        VkCommandBuffer* commandBuffer,
        VkDevice const& device,
        VkQueue const& queue
    );

    void* map(VkDevice& device, VkDeviceMemory& bufferMemory);
}