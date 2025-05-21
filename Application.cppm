module;

// Включаем динамическую библиотеку Vulkan
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC  1
#define VULKAN_HPP_NO_CONSTRUCTORS
#define VK_USE_PLATFORM_WIN32_KHR
#include <Windows.h>
#include <iostream>
#include <vulkan/vulkan.hpp>
#include <fstream>

export module Application;

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace VulkanSamples {

export struct vertex {
    float pos[3];
    float color[4];
};

export class Application {
public:
    Application(){}
    ~Application(){}

    void run(){

    // Вершины для прямоугольника
    vertex vertices[] = {
        -0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,
        0.5f, -0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f,
        0.5f, 0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f,
        -0.5f, 0.5f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f
    };

    // Индексы для прямоугольника
    uint32_t indices[] = {
        0, 1, 2,
        2, 3, 0
    };

    // Создаём класс для окна win32
    HINSTANCE hInstance = GetModuleHandle(nullptr);
    WNDCLASSW wc{};
    {
        wc.style = CS_HREDRAW | CS_VREDRAW;
        wc.lpfnWndProc = [](HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) -> LRESULT {
            switch (msg) {
            case WM_CLOSE:
                DestroyWindow(hwnd);
                std::cout << "WM_CLOSE" << std::endl;
                break;
            case WM_DESTROY:
                PostQuitMessage(0);
                std::cout << "WM_DESTROY" << std::endl;
                break;
            default:
                return DefWindowProc(hwnd, msg, wParam, lParam);
            }
            return DefWindowProc(hwnd, msg, wParam, lParam);
        };
        wc.hInstance = hInstance;
        wc.lpszClassName = L"VulkanSamples";
        wc.hIcon = LoadIcon(nullptr, IDI_APPLICATION);
        wc.hbrBackground = (HBRUSH)(COLOR_WINDOW+1);
        wc.cbClsExtra = 0;
        wc.cbWndExtra = 0;
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    }
    RegisterClassW(&wc);

    // Создаём окно win32
    HWND hwnd = CreateWindowW(L"VulkanSamples", L"VulkanSamples", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 800, 600, nullptr, nullptr, GetModuleHandle(nullptr), nullptr);

    VULKAN_HPP_DEFAULT_DISPATCHER.init();

    auto&& requiredInstanceExtensions = std::vector<const char*>{
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME
    };

    auto&& requiredInstanceLayers = std::vector<const char*>{
        "VK_LAYER_KHRONOS_validation"
    };


    vk::ApplicationInfo appInfo{
        .pApplicationName = "VulkanSamples",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "VulkanSamples",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_0
    };

    vk::StructureChain<vk::InstanceCreateInfo, vk::DebugUtilsMessengerCreateInfoEXT> instanceCreateInfo{
        vk::InstanceCreateInfo{
            .pApplicationInfo = &appInfo,
            .enabledLayerCount = (uint32_t)requiredInstanceLayers.size(),
            .ppEnabledLayerNames = requiredInstanceLayers.data(),
            .enabledExtensionCount = (uint32_t)requiredInstanceExtensions.size(),
            .ppEnabledExtensionNames = requiredInstanceExtensions.data(),
        },
        vk::DebugUtilsMessengerCreateInfoEXT{
            .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
            .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
            .pfnUserCallback = [] (vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageType, const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, void* userData) -> vk::Bool32 {
                switch(messageSeverity){
                    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose:
                        std::cout << "Verbose: ";
                        break;
                    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning:
                        std::cout << "Warning: ";
                        break;
                    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eError:
                        std::cout << "Error: ";
                        break;
                    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo:
                        std::cout << "Info: ";
                        break;
                }
                if(messageType & vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral) std::cout << "General: ";
                if(messageType & vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation) std::cout << "Validation: ";
                if(messageType & vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance) std::cout << "Performance: ";
                std::cout << pCallbackData->pMessage << std::endl;
                return VK_FALSE;
            },
            .pUserData = nullptr,
        },
    };

    vk::UniqueInstance instance = vk::createInstanceUnique(instanceCreateInfo.get<vk::InstanceCreateInfo>());
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance.get());

    vk::UniqueSurfaceKHR surface = instance->createWin32SurfaceKHRUnique(vk::Win32SurfaceCreateInfoKHR{
        .hinstance = hInstance,
        .hwnd = hwnd
    });

    vk::PhysicalDevice physicalDevice = instance->enumeratePhysicalDevices().front();

    std::vector<const char*> requiredDeviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

    // Выбираем очередь для отрисовки и показа
    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos{};
    float queuePriorities = 1.0f;
    float queuePriorities2[] = {1.0f, 1.0f};
    std::vector<uint32_t> graphicFamilyQueueIndices{};
    std::vector<uint32_t> presentFamilyQueueIndices{};

    for(uint32_t i = 0; i < queueFamilyProperties.size(); i++){
        auto&& queueFamilyProperty = queueFamilyProperties[i];
        if(queueFamilyProperty.queueFlags & vk::QueueFlagBits::eGraphics){
            graphicFamilyQueueIndices.push_back(i);
        }
        if(queueFamilyProperty.queueFlags & vk::QueueFlagBits::eTransfer){
            presentFamilyQueueIndices.push_back(i);
        }
    }

    // Заводим структуру для хранения индексов для очередей
    struct QueueIndices{
        std::pair<uint32_t, uint32_t> graphics;
        std::pair<uint32_t, uint32_t> present;
    } queueIndices;

    // Если есть семейство очередей, поддерживающее и показ, и отрисовку, то используем их
    for(auto&& graphicFamilyQueueIndex : graphicFamilyQueueIndices){
        for(auto&& presentFamilyQueueIndex : presentFamilyQueueIndices){
            if(graphicFamilyQueueIndex == presentFamilyQueueIndex){
                queueCreateInfos.push_back(vk::DeviceQueueCreateInfo{
                    .queueFamilyIndex = graphicFamilyQueueIndex,
                    .queueCount = 2,
                    .pQueuePriorities = queuePriorities2
                });
                queueIndices.graphics = std::make_pair(graphicFamilyQueueIndex, 0);
                queueIndices.present = std::make_pair(presentFamilyQueueIndex, 1);
                break;
            }
        }
        if(!queueCreateInfos.empty()) break;
    }

    // Если нет, то используем разные семейства очередей
    if(queueCreateInfos.empty()){
        queueIndices.graphics = std::make_pair(graphicFamilyQueueIndices.front(), 0);
        queueIndices.present = std::make_pair(presentFamilyQueueIndices.front(), 0);
        queueCreateInfos.push_back(vk::DeviceQueueCreateInfo{
            .queueFamilyIndex = queueIndices.graphics.first,
            .queueCount = 1,
            .pQueuePriorities = &queuePriorities
        });
        queueCreateInfos.push_back(vk::DeviceQueueCreateInfo{
            .queueFamilyIndex = queueIndices.present.first,
            .queueCount = 1,
            .pQueuePriorities = &queuePriorities
        });
    }

    vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice.getMemoryProperties();

    uint32_t memoryTypeIndex = 0;

    for(uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++){
        auto&& memoryType = memoryProperties.memoryTypes[i];
        if(memoryType.propertyFlags & (vk::MemoryPropertyFlagBits::eHostVisible)){
            memoryTypeIndex = i;
            break;
        }
    }

    vk::DeviceCreateInfo deviceCreateInfo{
        .queueCreateInfoCount = (uint32_t)queueCreateInfos.size(),
        .pQueueCreateInfos = queueCreateInfos.data(),
        .enabledExtensionCount = (uint32_t)requiredDeviceExtensions.size(),
        .ppEnabledExtensionNames = requiredDeviceExtensions.data()
    };

    vk::UniqueDevice device = physicalDevice.createDeviceUnique(deviceCreateInfo);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(device.get());

    // Получаем информацию о форматах показа
    vk::SurfaceFormatKHR surfaceFormat = physicalDevice.getSurfaceFormatsKHR(surface.get()).front();
    vk::SurfaceCapabilitiesKHR surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface.get());
    vk::UniqueSwapchainKHR swapchain = device->createSwapchainKHRUnique(vk::SwapchainCreateInfoKHR{
        .surface = surface.get(),
        .minImageCount = surfaceCapabilities.maxImageCount > 1 ? 2u : 1u,
        .imageFormat = surfaceFormat.format,
        .imageColorSpace = surfaceFormat.colorSpace,
        .imageExtent = surfaceCapabilities.currentExtent,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
        .imageSharingMode = vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .preTransform = surfaceCapabilities.currentTransform,
        .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
        .presentMode = vk::PresentModeKHR::eFifo,
        .clipped = true,
        .oldSwapchain = nullptr
    });

    std::vector<vk::Image> swapchainImages = device->getSwapchainImagesKHR(swapchain.get());
    std::vector<vk::UniqueImageView> imageViews{};

    for(vk::Image image : swapchainImages){
        vk::UniqueImageView imageView = device->createImageViewUnique(vk::ImageViewCreateInfo{
            .image = image,
            .viewType = vk::ImageViewType::e2D,
            .format = surfaceFormat.format,
            .subresourceRange = vk::ImageSubresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        });
        imageViews.push_back(std::move(imageView));
    }

    vk::UniqueBuffer vertextBuffer = device->createBufferUnique(vk::BufferCreateInfo{
        .size = sizeof(vertices),
        .usage = vk::BufferUsageFlagBits::eVertexBuffer,
        .sharingMode = vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr
    });

    vk::UniqueBuffer indexBuffer = device->createBufferUnique(vk::BufferCreateInfo{
        .size = sizeof(indices),
        .usage = vk::BufferUsageFlagBits::eIndexBuffer,
        .sharingMode = vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr
    });

    vk::DeviceSize memorySize = device->getBufferMemoryRequirements(vertextBuffer.get()).size + device->getBufferMemoryRequirements(indexBuffer.get()).size;

    vk::UniqueDeviceMemory memory = device->allocateMemoryUnique(vk::MemoryAllocateInfo{
        .allocationSize = memorySize,
        .memoryTypeIndex = memoryTypeIndex
    });

    device->bindBufferMemory(vertextBuffer.get(), memory.get(), 0);
    device->bindBufferMemory(indexBuffer.get(), memory.get(), device->getBufferMemoryRequirements(vertextBuffer.get()).size);

    void* mappedMemory = device->mapMemory(memory.get(), 0, memorySize);

    memcpy(mappedMemory, vertices, sizeof(vertices));
    memcpy((char*)mappedMemory + sizeof(vertices), indices, sizeof(indices));

    device->unmapMemory(memory.get());

    vk::AttachmentDescription colorAttachment = {
        .format = surfaceFormat.format,
        .samples = vk::SampleCountFlagBits::e1,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
        .initialLayout = vk::ImageLayout::eUndefined,
        .finalLayout = vk::ImageLayout::ePresentSrcKHR
    };

    vk::AttachmentReference colorAttachmentRef = {
        .attachment = 0,
        .layout = vk::ImageLayout::eColorAttachmentOptimal
    };

    vk::SubpassDescription subpass = {
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachmentRef
    };

    // Создадим RenderPass
    vk::UniqueRenderPass renderPass = device->createRenderPassUnique(vk::RenderPassCreateInfo{
        .attachmentCount = 1,
        .pAttachments = &colorAttachment,
        .subpassCount = 1,
        .pSubpasses = &subpass
    });
    
    // Создадим Framebuffer
    vk::UniqueFramebuffer framebuffers[] = {{
        device->createFramebufferUnique(vk::FramebufferCreateInfo{
            .renderPass = renderPass.get(),
            .attachmentCount = 1,
            .pAttachments = &imageViews[0].get(),
            .width = surfaceCapabilities.currentExtent.width,
            .height = surfaceCapabilities.currentExtent.height,
            .layers = 1
        })}, {
        device->createFramebufferUnique(vk::FramebufferCreateInfo{
            .renderPass = renderPass.get(),
            .attachmentCount = 1,
            .pAttachments = &imageViews[1].get(),
            .width = surfaceCapabilities.currentExtent.width,
            .height = surfaceCapabilities.currentExtent.height,
            .layers = 1
        })
        }
    };

    // Создадим PipelineLayout
    vk::UniquePipelineLayout pipelineLayout = device->createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo{
        .setLayoutCount = 0,
        .pSetLayouts = nullptr,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = nullptr
    });

    vk::UniquePipelineCache pipelineCache = device->createPipelineCacheUnique(vk::PipelineCacheCreateInfo{
        .initialDataSize = 0,
        .pInitialData = nullptr
    });

    std::ifstream vertexShaderFile("shaders/vert.spv", std::ios::binary);
    std::vector<char> vertexShader(std::istreambuf_iterator<char>(vertexShaderFile), {});

    std::ifstream fragmentShaderFile("shaders/frag.spv", std::ios::binary);
    std::vector<char> fragmentShader(std::istreambuf_iterator<char>(fragmentShaderFile), {});

    vk::UniqueShaderModule vertexShaderModule = device->createShaderModuleUnique(vk::ShaderModuleCreateInfo{
        .codeSize = vertexShader.size(),
        .pCode = reinterpret_cast<const uint32_t*>(vertexShader.data())
    });

    vk::UniqueShaderModule fragmentShaderModule = device->createShaderModuleUnique(vk::ShaderModuleCreateInfo{
        .codeSize = fragmentShader.size(),
        .pCode = reinterpret_cast<const uint32_t*>(fragmentShader.data())
    });

    vk::PipelineShaderStageCreateInfo shaderStages[2] = {
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = vertexShaderModule.get(),
            .pName = "main"
        }, 
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = fragmentShaderModule.get(),
            .pName = "main"
        }       
    };

    vk::VertexInputBindingDescription vertexInputBinding = {
        .binding = 0,
        .stride = 7 * sizeof(float),
        .inputRate = vk::VertexInputRate::eVertex
    };

    vk::VertexInputAttributeDescription vertexInputAttributes[2] = {
        // position
        vk::VertexInputAttributeDescription{
            .location = 0,
            .binding = 0,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = 0
        },
        // color
        vk::VertexInputAttributeDescription{
            .location = 1,
            .binding = 0,
            .format = vk::Format::eR32G32B32A32Sfloat,
            .offset = 3 * sizeof(float)
        }
    };

    vk::PipelineVertexInputStateCreateInfo vertexInputState = {
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &vertexInputBinding,
        .vertexAttributeDescriptionCount = 2,
        .pVertexAttributeDescriptions = vertexInputAttributes
    };

    vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState = {
        .topology = vk::PrimitiveTopology::eTriangleList,
        .primitiveRestartEnable = false
    };

    vk::Viewport viewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width = (float)surfaceCapabilities.currentExtent.width,
        .height = (float)surfaceCapabilities.currentExtent.height,
        .minDepth = 0.0f,
        .maxDepth = 1.0f
    };

    vk::Rect2D scissor = {
        .offset = {0, 0},
        .extent = surfaceCapabilities.currentExtent
    };

    vk::PipelineViewportStateCreateInfo viewportState = {
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor
    };

    vk::PipelineRasterizationStateCreateInfo rasterizationState = {
        .depthClampEnable = false,
        .rasterizerDiscardEnable = false,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eBack,
        .frontFace = vk::FrontFace::eClockwise,
        .depthBiasEnable = false,
        .lineWidth = 1.0f
    };

    vk::PipelineMultisampleStateCreateInfo multisampleState = {
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
        .sampleShadingEnable = false
    };

    vk::PipelineColorBlendAttachmentState colorBlendAttachment = {
        .blendEnable = false,
        .srcColorBlendFactor = vk::BlendFactor::eOne,
        .dstColorBlendFactor = vk::BlendFactor::eZero,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eOne,
        .dstAlphaBlendFactor = vk::BlendFactor::eZero,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
    };

    vk::PipelineColorBlendStateCreateInfo colorBlendState = {
        .logicOpEnable = false,
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachment
    };

    vk::UniquePipeline pipeline = device->createGraphicsPipelineUnique(pipelineCache.get(), vk::GraphicsPipelineCreateInfo{
        .stageCount = 2,
        .pStages = shaderStages,
        .pVertexInputState = &vertexInputState,
        .pInputAssemblyState = &inputAssemblyState,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizationState,
        .pMultisampleState = &multisampleState,
        .pDepthStencilState = nullptr,
        .pColorBlendState = &colorBlendState,
        .pDynamicState = nullptr,
        .layout = pipelineLayout.get(),
        .renderPass = renderPass.get(),
        .subpass = 0
    }).value;

    vk::Queue graphicQueue = device->getQueue(queueIndices.graphics.first, queueIndices.graphics.second);
    vk::Queue presentQueue = device->getQueue(queueIndices.present.first, queueIndices.present.second);

    vk::UniqueCommandPool commandPool = device->createCommandPoolUnique(vk::CommandPoolCreateInfo{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = queueCreateInfos[0].queueFamilyIndex
    });

    vk::CommandBuffer commandBuffer = device->allocateCommandBuffers(vk::CommandBufferAllocateInfo{
        .commandPool = commandPool.get(),
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1
    }).front();

    // create semaphores for synchronization
    vk::UniqueSemaphore imageAvailableSemaphore = device->createSemaphoreUnique(vk::SemaphoreCreateInfo{});
    vk::UniqueSemaphore renderFinishedSemaphore = device->createSemaphoreUnique(vk::SemaphoreCreateInfo{});
    vk::UniqueFence fence = device->createFenceUnique(vk::FenceCreateInfo{
        .flags = vk::FenceCreateFlagBits::eSignaled
    });
    vk::UniqueFence fence2 = device->createFenceUnique(vk::FenceCreateInfo{});

    vk::ClearValue clearValue {
        .color = std::array{1.0f, 0.5f, 0.0f, 1.0f}
    };

    ShowWindow(hwnd, SW_SHOWNORMAL);

    MSG msg{};


    while(true){// Обработка сообщений Windows
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            if (msg.message == WM_QUIT) {
                break;
            }
            // 1. Ожидаем завершения предыдущего кадра (используем fence)
            vk::Result waitResult = device->waitForFences(fence.get(), VK_TRUE, UINT64_MAX);
            if (waitResult != vk::Result::eSuccess) {
                std::cerr << "Ошибка при ожидании fence: " << vk::to_string(waitResult) << std::endl;
            }
            device->resetFences(fence.get());
            // 2. Получаем индекс следующего изображения в цепочке обмена
            vk::ResultValue<uint32_t> acquireResult = device->acquireNextImageKHR(
                swapchain.get(), UINT64_MAX, imageAvailableSemaphore.get(), nullptr);

            uint32_t imageIndex = acquireResult.value;

            if(acquireResult.result != vk::Result::eSuccess){
                std::cerr << "Ошибка при получении следующего изображения: " << vk::to_string(acquireResult.result) << std::endl;
            }

            // 3. Записываем команды в командный буфер
            commandBuffer.reset(vk::CommandBufferResetFlagBits::eReleaseResources);

            commandBuffer.begin(vk::CommandBufferBeginInfo{
                .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
            });

            // Начинаем проход рендеринга
            vk::RenderPassBeginInfo renderPassInfo{
                .renderPass = renderPass.get(),
                .framebuffer = framebuffers[imageIndex].get(),
                .renderArea = vk::Rect2D{
                    .offset = vk::Offset2D{0, 0},
                    .extent = surfaceCapabilities.currentExtent
                },
                .clearValueCount = 1,
                .pClearValues = &clearValue //  Убедитесь, что clearValue инициализирован!
            };
            commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

            // Привязываем ресурсы и выполняем отрисовку
            commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.get());
            commandBuffer.bindVertexBuffers(0, vertextBuffer.get(), vk::DeviceSize(0));
            commandBuffer.bindIndexBuffer(indexBuffer.get(), 0, vk::IndexType::eUint32);
            commandBuffer.drawIndexed(6, 1, 0, 0, 0);

            // Завершаем проход рендеринга
            commandBuffer.endRenderPass();
            commandBuffer.end();

            // 4. Отправляем командный буфер в графическую очередь
            vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
            vk::SubmitInfo submitInfo{
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = &imageAvailableSemaphore.get(),
                .pWaitDstStageMask = waitStages,
                .commandBufferCount = 1,
                .pCommandBuffers = &commandBuffer,
                .signalSemaphoreCount = 1,
                .pSignalSemaphores = &renderFinishedSemaphore.get()
            };
            graphicQueue.submit(submitInfo, fence.get()); // ИСПОЛЬЗУЕМ FENCE!


            // 5. Представляем изображение
            vk::PresentInfoKHR presentInfo{
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = &renderFinishedSemaphore.get(),
                .swapchainCount = 1,
                .pSwapchains = &swapchain.get(),
                .pImageIndices = &imageIndex
            };

            vk::Result presentResult = presentQueue.presentKHR(presentInfo);
        }
    }
    }

private:
    VkInstance instance;
};

}