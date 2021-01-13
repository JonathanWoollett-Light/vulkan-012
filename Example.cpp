#include "Example.hpp"

#include <filesystem>

ComputeApp::ComputeApp(
    char const* shaderFile,
    uint32_t const bufferSize,
    float* bufferData,
    uint32_t* dims, // [x,y,z],
    uint32_t const* dimLengths // [local_size_x, local_size_y, local_size_z]
) {
    // Initialize vulkan:
    Utility::createInstance(this->instance);

    // Gets physical device
    Utility::getPhysicalDevice(this->instance, this->physicalDevice);

    // Gets logical device
    Utility::createDevice(this->physicalDevice, this->queueFamilyIndex, this->device, this->queue);

    // Creates buffers
    Utility::createBuffer(this->physicalDevice, this->device, bufferSize, &this->buffer, &this->bufferMemory);

    // Fills buffers
    Utility::fillBuffer(this->device, this->bufferMemory, bufferData, bufferSize);

    // Creates descriptor set layout
    Utility::createDescriptorSetLayout(this->device,&this->descriptorSetLayout);

    // Creates descriptor set
    Utility::createDescriptorSet(this->device,&this->descriptorPool,&this->descriptorSetLayout,buffer,this->descriptorSet);

    // Creates compute pipeline
    Utility::createComputePipeline(
        this->device,
        shaderFile,
        &this->computeShaderModule,
        &this->descriptorSetLayout,
        &this->pipelineLayout,
        &this->pipeline
    );

    // Creates command buffer
    Utility::createCommandBuffer(
        this->queueFamilyIndex,
        this->device,
        &this->commandPool,
        &this->commandBuffer,
        this->pipeline,
        this->pipelineLayout,
        this->descriptorSet,
        dims,
        dimLengths
    );

    Utility::runCommandBuffer(
        &this->commandBuffer,
        this->device,
        this->queue
    );
}

ComputeApp::~ComputeApp() {
    vkFreeMemory(device, bufferMemory, nullptr);
    vkDestroyBuffer(device, buffer, nullptr);

    // TODO STRUCT DESCTUROR
    vkDestroyShaderModule(device, computeShaderModule, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyCommandPool(device, commandPool, nullptr);

    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);		
}

// Gets Vulkan instance
void Utility::createInstance(VkInstance& instance) {
    std::vector<char const*> enabledLayers;
    std::vector<char const*> enabledExtensions;

    if (enableValidationLayers.has_value()) {
        // Gets number of supported layers
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        // Gets all supported layers
        std::vector<VkLayerProperties> layerProperties(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, layerProperties.data());

        // Check 'VK_LAYER_KHRONOS_validation' is among supported layers
        auto layer_itr = std::find_if(layerProperties.begin(), layerProperties.end(), [](VkLayerProperties& prop) {
            return (strcmp(enableValidationLayers.value(), prop.layerName) == 0);
        });
        // If not, throw error
        if (layer_itr == layerProperties.end()) {
            throw std::runtime_error("Validation layer not supported\n");
        }
        // Else, push to layers
        enabledLayers.push_back(enableValidationLayers.value());

        // We need to enable the extension VK_EXT_DEBUG_REPORT,
        //  to print the warnings emitted by the validation layer.

        // Gets number of supported extensions
        uint32_t extensionCount;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
        // Gets all supported extensions
        std::vector<VkExtensionProperties> extensionProperties(extensionCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensionProperties.data());

        // Check `VK_EXT_DEBUG_REPORT` is among supported extensions
        bool debug_report = false;
        for(VkExtensionProperties& prop: extensionProperties) {
            if (strcmp(VK_EXT_DEBUG_REPORT_EXTENSION_NAME, prop.extensionName) == 0) { debug_report=true; break; }
        }
        // If not, throw error
        if (!debug_report) {
            throw std::runtime_error("Extension VK_EXT_DEBUG_REPORT_EXTENSION_NAME not supported\n");
        }
        // Else, push to extensions
        enabledExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
    }

    VkInstanceCreateInfo createInfo = {};
    {
        VkApplicationInfo applicationInfo = {};
        {
            applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
            applicationInfo.apiVersion = VK_API_VERSION_1_1;
        }
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &applicationInfo;
        createInfo.enabledLayerCount = static_cast<uint32_t>(enabledLayers.size());
        createInfo.ppEnabledLayerNames = enabledLayers.data();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size());
        createInfo.ppEnabledExtensionNames = enabledExtensions.data();
    }

    // Creates instance
    #pragma warning(disable : 26812) // Removes enum scoping warnings from Vulkan
    VK_CHECK_RESULT(vkCreateInstance(
        &createInfo,
        nullptr,
        &instance)
    );
    #pragma warning(default : 26812)
}

// Gets physical device
void Utility::getPhysicalDevice(VkInstance const& instance, VkPhysicalDevice& physicalDevice) {
    // Gets number of physical devices
    uint32_t deviceCount;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    // Asserts a system has a device
    assert(deviceCount != 0);

    // Gets physical devices
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    // Picks 1st
    physicalDevice = devices[0];
}

// Gets index of 1st queue family which supports compute
uint32_t Utility::getComputeQueueFamilyIndex(VkPhysicalDevice const& physicalDevice) {
    // Gets number of queue families
    uint32_t queueFamilyCount;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);

    // Gets queue families
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

    // Finds 1st queue family which supports compute
    auto itr = std::find_if(queueFamilies.begin(), queueFamilies.end(), [](VkQueueFamilyProperties& props) {
        return (props.queueFlags & VK_QUEUE_COMPUTE_BIT);
    });
    if (itr == queueFamilies.end()) {
        throw std::runtime_error("No compute queue family");
    }
    return std::distance(queueFamilies.begin(), itr);
}

// Creates logical device
void Utility::createDevice(
    VkPhysicalDevice const& physicalDevice,
    uint32_t& queueFamilyIndex,
    VkDevice& device,
    VkQueue& queue
) {
    // Device info
    VkDeviceCreateInfo deviceCreateInfo = {};
    {   
        // Device queue info
        VkDeviceQueueCreateInfo queueCreateInfo = {};
        {
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueFamilyIndex = getComputeQueueFamilyIndex(physicalDevice); // find queue family with compute capability.
            queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
            queueCreateInfo.queueCount = 1; // create one queue in this family. We don't need more.
            queueCreateInfo.pQueuePriorities = new float(1.0); // For 1 queue we simply need a 1 value array of the priority
        }
        deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
        deviceCreateInfo.queueCreateInfoCount = 1;
    }

    VK_CHECK_RESULT(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device)); // create logical device.

    // Get handle to queue 0 in `queueFamilyIndex` queue family
    vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
}

// Finds memory type with given properties
uint32_t Utility::findMemoryType(VkPhysicalDevice const& physicalDevice, uint32_t const memoryTypeBits, VkMemoryPropertyFlags const properties) {
    VkPhysicalDeviceMemoryProperties memoryProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    // We iterate through the memory types we can allocate from heaps of this physical device

   // `memoryProperties::memoryType` is an array of `VkMemorytype`s
   //  describing the memory types that can be used to access memory
   //  allocated from the heaps specified by `memoryHeaps`
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        if ((memoryTypeBits & (1 << i)) &&
            ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties))
            return i;
    }
    return -1;
}

// Creates buffer
void Utility::createBuffer(
    VkPhysicalDevice const& physicalDevice,
    VkDevice const& device,
    uint32_t const size,
    VkBuffer * const buffer,
    VkDeviceMemory * const bufferMemory
) {
    // Buffer info
    VkBufferCreateInfo bufferCreateInfo = {};
    {
        bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCreateInfo.size = sizeof(float)*size; // buffer size in bytes. 
        bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // buffer is used as a storage buffer.
        bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 
    }

    VK_CHECK_RESULT(vkCreateBuffer(device, &bufferCreateInfo, nullptr, buffer)); // Constructs buffer

    // Buffers do not allocate memory upon construction, we must do it manually
    
    // Gets buffer memory size and offset
    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(device, *buffer, &memoryRequirements);
    
    // Sets buffer options
    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize = memoryRequirements.size; // Bytes

    allocateInfo.memoryTypeIndex = findMemoryType(
        physicalDevice,
        // Sets memory must supports all the operations our buffer memory supports
        memoryRequirements.memoryTypeBits,
        // Sets memory must have the properties:
        //  `VK_MEMORY_PROPERTY_HOST_COHERENT_BIT` Can more easily view
        //  `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` Can read from GPU to CPU
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
    );

    // Allocates memory
    VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfo, nullptr, bufferMemory));

    // Binds buffer to allocated memory
    VK_CHECK_RESULT(vkBindBufferMemory(device, *buffer, *bufferMemory, 0));
}

// Fills buffers with data
// Must be after `createBuffers` but before `createComputePipeline`
void Utility::fillBuffer(
    VkDevice const& device,
    VkDeviceMemory& bufferMemory,
    float*& bufferData, 
    uint32_t const bufferSize
) {
    void* data = nullptr;
    vkMapMemory(device, bufferMemory, 0, VK_WHOLE_SIZE, 0, &data);
    memcpy(data, bufferData, (size_t)(bufferSize * sizeof(float)));
    vkUnmapMemory(device, bufferMemory);
}

// Creates descriptor set layout
void Utility::createDescriptorSetLayout(
    VkDevice const& device,
    VkDescriptorSetLayout* descriptorSetLayout
) {
    // Descriptor set layout options
    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
    {
        VkDescriptorSetLayoutBinding binding = {};
        {
            binding.binding = 0; // `layout(binding = i)`
            binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binding.descriptorCount = 1; // TODO Wtf does this do? Number of items in buffer maybe?
            binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        
        descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetLayoutCreateInfo.bindingCount = 1; // 1 descriptor/bindings in this descriptor set
        descriptorSetLayoutCreateInfo.pBindings = &binding;
    }
    
    // Create the descriptor set layout. 
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, nullptr, descriptorSetLayout));
}

// Creates descriptor set
void Utility::createDescriptorSet(
    VkDevice const& device,
    VkDescriptorPool* descriptorPool,
    VkDescriptorSetLayout* descriptorSetLayout,
    VkBuffer& buffer,
    VkDescriptorSet& descriptorSet
) {
    // Creates descriptor pool
    // A pool implements a number of descriptors of each type 
    //  `VkDescriptorPoolSize` specifies for each descriptor type the number to hold
    // A descriptor set is initialised to contain all descriptors defined in a descriptor pool
    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
    {
        // Descriptor type and number
        VkDescriptorPoolSize descriptorPoolSize = {};
        {
            descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // Descriptor type
            descriptorPoolSize.descriptorCount = 1; // Number of descriptors
        }
        descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptorPoolCreateInfo.maxSets = 1; // max number of sets that can be allocated from this pool
        descriptorPoolCreateInfo.poolSizeCount = 1; // length of `pPoolSizes`
        descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize; // pointer to array of `VkDescriptorPoolSize`
    }

    // create descriptor pool.
    VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, nullptr, descriptorPool));

    // Specifies options for creation of multiple of descriptor sets
    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
    {
        descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocateInfo.descriptorPool = *descriptorPool; // pool from which sets will be allocated
        descriptorSetAllocateInfo.descriptorSetCount = 1; // number of descriptor sets to implement (also length of `pSetLayouts`)
        descriptorSetAllocateInfo.pSetLayouts = descriptorSetLayout; // pointer to array of descriptor set layouts
    }
    

    // allocate descriptor set.
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, &descriptorSet));

    // Gets buffer memory size and offset
    // VkMemoryRequirements memoryRequirements;
    // vkGetBufferMemoryRequirements(device, *buffer, &memoryRequirements);

    // Binds descriptors from our descriptor sets to our buffers
    VkWriteDescriptorSet writeDescriptorSet = {};
    {
        // Binds descriptors to buffers
        VkDescriptorBufferInfo binding = {};
        {
            binding.buffer = buffer;
            binding.offset = 0;
            // If size not given, set to whole size of buffer
            binding.range = VK_WHOLE_SIZE;
        }

        writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSet.dstSet = descriptorSet; // write to this descriptor set.
        // TODO Wtf does this do?
        //  My best guess is that descriptor sets can have multile bindings to different sets of buffers.
        //  original comment said 'write to the first, and only binding.'
        writeDescriptorSet.dstBinding = 0;
        writeDescriptorSet.descriptorCount = 1; // update all descriptors in set.
        writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // storage buffer.
        writeDescriptorSet.pBufferInfo = &binding;
    }
    
    // perform the update of the descriptor set.
    vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
}

// Reads shader file
uint32_t* Utility::readShader(uint32_t& length, char const* filename) {
    // std::string path = "../../../";
    // std::cout << "paths:" << std::endl;
    // for (const auto & entry : std::filesystem::directory_iterator(path)) {
    //     std::cout << entry.path() << std::endl;
    // }
    
    // Open file
    FILE* fp = fopen(filename, "rb");
    if (fp == nullptr) {
        throw std::runtime_error("Could not open shader");
    }

    // Get file size.
    fseek(fp, 0, SEEK_END);
    long filesize = ftell(fp);
    // Get file size ceiled to multiple of 4 bytes
    fseek(fp, 0, SEEK_SET);
    long filesizepadded = long(ceil(filesize / 4.0)) * 4;

    // Read file
    char *str = new char[filesizepadded];
    fread(str, filesizepadded, sizeof(char), fp);

    // Close file
    fclose(fp);

    // zeros last 0 to 3 bytes
    for (int i = filesize; i < filesizepadded; i++) {
        str[i] = 0;
    }

    length = filesizepadded;
    return (uint32_t *)str;
}

// Creates compute pipeline
void Utility::createComputePipeline(
    VkDevice const& device,
    char const* shaderFile,
    VkShaderModule* computeShaderModule,
    VkDescriptorSetLayout* descriptorSetLayout,
    VkPipelineLayout* pipelineLayout,
    VkPipeline* pipeline
) {
    // Creates shader module (just a wrapper around our shader)
    VkShaderModuleCreateInfo createInfo = {};
    {
        uint32_t filelength;
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.pCode = readShader(filelength, shaderFile);
        createInfo.codeSize = filelength;
    }

    VK_CHECK_RESULT(vkCreateShaderModule(device, &createInfo, nullptr, computeShaderModule));

    // A compute pipeline is very simple compared to a graphics pipeline.
    // It only consists of a single stage with a compute shader.

    // The pipeline layout allows the pipeline to access descriptor sets. 
    // So we just specify the descriptor set layout we created earlier.
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
    {
        pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutCreateInfo.setLayoutCount = 1; // 1 shader
        pipelineLayoutCreateInfo.pSetLayouts = descriptorSetLayout; // Descriptor set        
    }
    
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, pipelineLayout));

    // Set our pipeline options
    VkComputePipelineCreateInfo pipelineCreateInfo = {};
    {
        // We specify the compute shader stage, and it's entry point(main).
        VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
        {
            shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT; // Shader type
            shaderStageCreateInfo.module = *computeShaderModule; // Shader module
            shaderStageCreateInfo.pName = "main"; // Shader entry point
        }
        // We set our pipeline options
        pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineCreateInfo.stage = shaderStageCreateInfo; // Shader stage info
        pipelineCreateInfo.layout = *pipelineLayout;
    }
    

    // Create compute pipeline
    VK_CHECK_RESULT(vkCreateComputePipelines(
        device, VK_NULL_HANDLE,
        1, &pipelineCreateInfo,
        nullptr, pipeline));
}

// Creates command buffer
void Utility::createCommandBuffer(
    uint32_t queueFamilyIndex,
    VkDevice& device,
    VkCommandPool* commandPool,
    VkCommandBuffer* commandBuffer,
    VkPipeline& pipeline,
    VkPipelineLayout& pipelineLayout,
    VkDescriptorSet& descriptorSet,
    uint32_t const* dims, // [x,y,z],
    uint32_t const* dimLengths // [local_size_x, local_size_y, local_size_z]
) {
    // Creates command pool
    VkCommandPoolCreateInfo commandPoolCreateInfo = {};
    {
        commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolCreateInfo.flags = 0;
        // Sets queue family
        commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;
    }
    VK_CHECK_RESULT(vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, commandPool));

    //  Allocates command buffer
    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
    {
        commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferAllocateInfo.commandPool = *commandPool; // Pool to allocate from
        commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferAllocateInfo.commandBufferCount = 1; // Allocates 1 command buffer. 
    }
    VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, commandBuffer)); // allocate command buffer.

    // Allocated command buffer options
    VkCommandBufferBeginInfo beginInfo = {};
    {
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        // Buffer only submitted once
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    }
    // Start recording commands
    VK_CHECK_RESULT(vkBeginCommandBuffer(*commandBuffer, &beginInfo));

    // Binds pipeline (our functions)
    vkCmdBindPipeline(*commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    // Binds descriptor set (our data)
    vkCmdBindDescriptorSets(*commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

    // Sets invocations
    vkCmdDispatch(
        *commandBuffer,
        ceil(dims[0] / static_cast<float>(dimLengths[0])),
        ceil(dims[1] / static_cast<float>(dimLengths[1])),
        ceil(dims[2] / static_cast<float>(dimLengths[2]))
    );

    // End recording commands
    VK_CHECK_RESULT(vkEndCommandBuffer(*commandBuffer));
}

// Runs command buffer
void Utility::runCommandBuffer(
    VkCommandBuffer* commandBuffer,
    VkDevice const& device,
    VkQueue const& queue
) {
    VkSubmitInfo submitInfo = {};
    {
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        // submit 1 command buffer
        submitInfo.commandBufferCount = 1;
        // pointer to array of command buffers to submit
        submitInfo.pCommandBuffers = commandBuffer;
    }

    // Creates fence (so we can await for command buffer to finish)
    VkFence fence;
    VkFenceCreateInfo fenceCreateInfo = {};
    {
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        // fenceCreateInfo.flags = 0; // this is set by default // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkFenceCreateFlagBits.html
    }
    VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, nullptr, &fence));

    // Submit command buffer with fence
    VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence));

    // Wait for fence to signal (which it does when command buffer has finished)
    VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000));

    // Destructs fence
    vkDestroyFence(device, fence, nullptr);
}

// Maps buffer to CPU memory
void* Utility::map(
    VkDevice& device,
    VkDeviceMemory& bufferMemory
) {
    void* data = nullptr;
    vkMapMemory(device, bufferMemory, 0, VK_WHOLE_SIZE, 0, &data);
    return data;
}