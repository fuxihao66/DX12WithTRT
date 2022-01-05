/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


// TODO: 如何share with cuda 
// 0. 创建share handle  （注意创建资源时heap type设置为share）
// 1. import to cuda  (cudaExternalMemoryDedicated这个flag需要设置）
// 2. map to cuda pointer
// TODO: 如何同步 
// 0. 创建fence的share handle
// 1. import fence object to cuda
// 2. 


// 需要include trt的头文件和common的头文件
// 并且把common中的logger.cpp添加到工程

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"
#include <windows.h>

#include "d3dx12.h"

#include <string>
#include <wrl.h>
#include <shellapi.h>

#include <cuda_runtime.h>
#include "ShaderStructs.h"
#include "simpleD3D12.h"
#include <aclapi.h>


//////////////////////////////////////////////
// WindowsSecurityAttributes implementation //
//////////////////////////////////////////////

class WindowsSecurityAttributes {
 protected:
  SECURITY_ATTRIBUTES m_winSecurityAttributes;
  PSECURITY_DESCRIPTOR m_winPSecurityDescriptor;

 public:
  WindowsSecurityAttributes();
  ~WindowsSecurityAttributes();
  SECURITY_ATTRIBUTES *operator&();
};

WindowsSecurityAttributes::WindowsSecurityAttributes() {
  m_winPSecurityDescriptor = (PSECURITY_DESCRIPTOR)calloc(
      1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void **));
  assert(m_winPSecurityDescriptor != (PSECURITY_DESCRIPTOR)NULL);

  PSID *ppSID = (PSID *)((PBYTE)m_winPSecurityDescriptor +
                         SECURITY_DESCRIPTOR_MIN_LENGTH);
  PACL *ppACL = (PACL *)((PBYTE)ppSID + sizeof(PSID *));

  InitializeSecurityDescriptor(m_winPSecurityDescriptor,
                               SECURITY_DESCRIPTOR_REVISION);

  SID_IDENTIFIER_AUTHORITY sidIdentifierAuthority =
      SECURITY_WORLD_SID_AUTHORITY;
  AllocateAndInitializeSid(&sidIdentifierAuthority, 1, SECURITY_WORLD_RID, 0, 0,
                           0, 0, 0, 0, 0, ppSID);

  EXPLICIT_ACCESS explicitAccess;
  ZeroMemory(&explicitAccess, sizeof(EXPLICIT_ACCESS));
  explicitAccess.grfAccessPermissions =
      STANDARD_RIGHTS_ALL | SPECIFIC_RIGHTS_ALL;
  explicitAccess.grfAccessMode = SET_ACCESS;
  explicitAccess.grfInheritance = INHERIT_ONLY;
  explicitAccess.Trustee.TrusteeForm = TRUSTEE_IS_SID;
  explicitAccess.Trustee.TrusteeType = TRUSTEE_IS_WELL_KNOWN_GROUP;
  explicitAccess.Trustee.ptstrName = (LPTSTR)*ppSID;

  SetEntriesInAcl(1, &explicitAccess, NULL, ppACL);

  SetSecurityDescriptorDacl(m_winPSecurityDescriptor, TRUE, *ppACL, FALSE);

  m_winSecurityAttributes.nLength = sizeof(m_winSecurityAttributes);
  m_winSecurityAttributes.lpSecurityDescriptor = m_winPSecurityDescriptor;
  m_winSecurityAttributes.bInheritHandle = TRUE;
}

WindowsSecurityAttributes::~WindowsSecurityAttributes() {
  PSID *ppSID = (PSID *)((PBYTE)m_winPSecurityDescriptor +
                         SECURITY_DESCRIPTOR_MIN_LENGTH);
  PACL *ppACL = (PACL *)((PBYTE)ppSID + sizeof(PSID *));

  if (*ppSID) {
    FreeSid(*ppSID);
  }
  if (*ppACL) {
    LocalFree(*ppACL);
  }
  free(m_winPSecurityDescriptor);
}

SECURITY_ATTRIBUTES *WindowsSecurityAttributes::operator&() {
  return &m_winSecurityAttributes;
}

DX12CudaInterop::DX12CudaInterop(UINT width, UINT height, std::string name)
    : DX12CudaSample(width, height, name),
      m_frameIndex(0),
      m_scissorRect(0, 0, static_cast<LONG>(width), static_cast<LONG>(height)),
      m_fenceValues{}{
  m_viewport = {0.0f, 0.0f, static_cast<float>(width),
                static_cast<float>(height)};
  m_AnimTime = 1.0f;
}

void DX12CudaInterop::OnInit() {
  LoadPipeline();
  InitCuda();
  LoadAssets();
}

// Load the rendering pipeline dependencies.
void DX12CudaInterop::LoadPipeline() {
  UINT dxgiFactoryFlags = 0;

#if defined(_DEBUG)
  // Enable the debug layer (requires the Graphics Tools "optional feature").
  // NOTE: Enabling the debug layer after device creation will invalidate the
  // active device.
  {
    ComPtr<ID3D12Debug> debugController;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
      debugController->EnableDebugLayer();

      // Enable additional debug layers.
      dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
    }
  }
#endif

  ComPtr<IDXGIFactory4> factory;
  ThrowIfFailed(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)));

  if (m_useWarpDevice) {
    ComPtr<IDXGIAdapter> warpAdapter;
    ThrowIfFailed(factory->EnumWarpAdapter(IID_PPV_ARGS(&warpAdapter)));

    ThrowIfFailed(D3D12CreateDevice(warpAdapter.Get(), D3D_FEATURE_LEVEL_12_0,
                                    IID_PPV_ARGS(&m_device)));
  } else {
    ComPtr<IDXGIAdapter1> hardwareAdapter;
    GetHardwareAdapter(factory.Get(), &hardwareAdapter);

    ThrowIfFailed(D3D12CreateDevice(hardwareAdapter.Get(),
                                    D3D_FEATURE_LEVEL_12_0,
                                    IID_PPV_ARGS(&m_device)));
    DXGI_ADAPTER_DESC1 desc;
    hardwareAdapter->GetDesc1(&desc);
    m_dx12deviceluid = desc.AdapterLuid;
  }

  // Describe and create the command queue.
  D3D12_COMMAND_QUEUE_DESC queueDesc = {};
  queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
  queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

  ThrowIfFailed(
      m_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_commandQueue)));

  // Describe and create the swap chain.
  DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
  swapChainDesc.BufferCount = FrameCount;
  swapChainDesc.Width = m_width;
  swapChainDesc.Height = m_height;
  //swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  swapChainDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
  swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
  swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
  swapChainDesc.SampleDesc.Count = 1;

  ComPtr<IDXGISwapChain1> swapChain;
  ThrowIfFailed(factory->CreateSwapChainForHwnd(
      m_commandQueue.Get(),  // Swap chain needs the queue so that it can force
                             // a flush on it.
      Win32Application::GetHwnd(), &swapChainDesc, nullptr, nullptr,
      &swapChain));

  // This sample does not support fullscreen transitions.
  ThrowIfFailed(factory->MakeWindowAssociation(Win32Application::GetHwnd(),
                                               DXGI_MWA_NO_ALT_ENTER));

  ThrowIfFailed(swapChain.As(&m_swapChain));
  m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

  //// Create descriptor heaps.
  //{
  //  // Describe and create a render target view (RTV) descriptor heap.
  //  D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
  //  rtvHeapDesc.NumDescriptors = FrameCount;
  //  rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
  //  rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
  //  ThrowIfFailed(
  //      m_device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&m_rtvHeap)));

  //  m_rtvDescriptorSize = m_device->GetDescriptorHandleIncrementSize(
  //      D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
  //}

  // Create frame resources.
  {
    /*CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(
        m_rtvHeap->GetCPUDescriptorHandleForHeapStart());*/

    // Create a RTV and a command allocator for each frame.
    for (UINT n = 0; n < FrameCount; n++) {
      ThrowIfFailed(
          m_swapChain->GetBuffer(n, IID_PPV_ARGS(&m_renderTargets[n])));
      /*m_device->CreateRenderTargetView(m_renderTargets[n].Get(), nullptr,
                                       rtvHandle);
      rtvHandle.Offset(1, m_rtvDescriptorSize);*/

      ThrowIfFailed(m_device->CreateCommandAllocator(
          D3D12_COMMAND_LIST_TYPE_DIRECT,
          IID_PPV_ARGS(&m_commandAllocators[n])));
    }



  }
}

void DX12CudaInterop::InitCuda() {
  int num_cuda_devices = 0;
  checkCudaErrors(cudaGetDeviceCount(&num_cuda_devices));

  if (!num_cuda_devices) {
    throw std::exception("No CUDA Devices found");
  }
  for (UINT devId = 0; devId < num_cuda_devices; devId++) {
    cudaDeviceProp devProp;
    checkCudaErrors(cudaGetDeviceProperties(&devProp, devId));

    if ((memcmp(&m_dx12deviceluid.LowPart, devProp.luid,
                sizeof(m_dx12deviceluid.LowPart)) == 0) &&
        (memcmp(&m_dx12deviceluid.HighPart,
                devProp.luid + sizeof(m_dx12deviceluid.LowPart),
                sizeof(m_dx12deviceluid.HighPart)) == 0)) {
      checkCudaErrors(cudaSetDevice(devId));
      m_cudaDeviceID = devId;
      m_nodeMask = devProp.luidDeviceNodeMask;
      checkCudaErrors(cudaStreamCreate(&m_streamToRun));
      printf("CUDA Device Used [%d] %s\n", devId, devProp.name);
      break;
    }
  }
}
// Load the sample assets.
void DX12CudaInterop::LoadAssets() {

    BuildComputePipelineAndRS();
    // Create the command list.
    ThrowIfFailed(m_commandAllocators[m_frameIndex]->Reset());

    ThrowIfFailed(m_device->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_DIRECT,
        m_commandAllocators[m_frameIndex].Get(), m_maskPipeline.Get(),
        IID_PPV_ARGS(&m_commandList)));

    PrepareResources();
    ThrowIfFailed(m_commandList->Close());

    ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
    m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);   // 资源初始化



    /*LoadONNXModel("H:/multiFrame测速/UNetGated.onnx");
    ShareTextureWithCuda();*/

    // TODO: 资源初始化之后需要同步 
    // Create synchronization objects and wait until assets have been uploaded to the GPU.
    {m_fenceValues[m_frameIndex] += 1;
        ThrowIfFailed(m_device->CreateFence(m_fenceValues[m_frameIndex], D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)));
        m_fenceValues[m_frameIndex]++;

        // Create an event handle to use for frame synchronization.
        m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (m_fenceEvent == nullptr)
        {
            ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
        }


        WaitForGpu();
    }
}

bool DX12CudaInterop::LoadONNXModel(const std::string& filePath) {
    samplesCommon::OnnxSampleParams params;
    {
        params.onnxFileName = filePath;
        std::vector<std::string> inputsNames = { "warp_no_hole", "warp_occ","normal", "mask_one_channel", "history_1","history_2", "history_3" };
        params.inputTensorNames = inputsNames;
        std::vector<std::string> outputsNames = { "output_1" };
        params.outputTensorNames = outputsNames;
        params.dlaCore = 0;    // TODO:  
        params.int8 = false;
        params.fp16 = true;
        
    }
    auto m_onnxInferEngine = std::make_shared<OnnxEngine>(params);
    m_onnxInferEngine->build();

    m_bufferManager = std::make_shared<NetworkBufferManager>(m_onnxInferEngine->GetEngine());

    return true;
}
std::shared_ptr<Texture> DX12CudaInterop::CreateTextureFromExr(const std::string& filePath, DXGI_FORMAT textureFormat, unsigned int width, unsigned int height) {

    auto tex = std::make_shared<Texture>();

    D3D12_RESOURCE_DESC textureDesc = {};
    textureDesc.MipLevels = 1;
    textureDesc.Format = textureFormat;
    textureDesc.Width = width;
    textureDesc.Height = height;
    textureDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
    textureDesc.DepthOrArraySize = 1;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.SampleDesc.Quality = 0;
    textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;

    ThrowIfFailed(m_device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_SHARED,
        &textureDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&tex->defaultBuffer)));



    D3D12_SUBRESOURCE_DATA subresources;
    subresources.RowPitch = 1280 * 4 * sizeof(float);  // width * byte per pixel
    subresources.SlicePitch = 1280 * 720 * 4 * sizeof(float);
    float* out; // width * height * RGBA
    int texWidth;
    int texHeight;
    const char* err = NULL; // or nullptr in C++11

    int ret = LoadEXR(&out, &texWidth, &texHeight, filePath.c_str(), &err);

    if (ret != TINYEXR_SUCCESS) {
        if (err) {
            fprintf(stderr, "ERR : %s\n", err);
            FreeEXRErrorMessage(err); // release memory of error message.
        }
    }
    else {
        subresources.pData = out;
    }

    UINT64 requiredSize = GetRequiredIntermediateSize(tex->defaultBuffer.Get(), 0, 1);


    ThrowIfFailed(m_device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(requiredSize),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&tex->uploadBuffer)));
    
    UpdateSubresources(m_commandList.Get(), tex->defaultBuffer.Get(), tex->uploadBuffer.Get(), 0, 0, 1, &subresources);


    m_commandList->ResourceBarrier(
        1, &CD3DX12_RESOURCE_BARRIER::Transition(
            tex->defaultBuffer.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));


    return tex;
}

std::shared_ptr<Texture> DX12CudaInterop::CreateDummy(DXGI_FORMAT textureFormat, D3D12_RESOURCE_STATES initState, unsigned int width, unsigned int height) {

    auto tex = std::make_shared<Texture>();

    D3D12_RESOURCE_DESC textureDesc = {};
    textureDesc.MipLevels = 1;
    textureDesc.Format = textureFormat;
    textureDesc.Width = width;
    textureDesc.Height = height;
    textureDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    textureDesc.DepthOrArraySize = 1;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.SampleDesc.Quality = 0;
    textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;

    ThrowIfFailed(m_device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_SHARED,   // TODO:   这里是必要的 设置为shared 
        &textureDesc,
        initState,
        nullptr,
        IID_PPV_ARGS(&tex->defaultBuffer)));

    return tex;
}
void DX12CudaInterop::PrepareResources() {
    const std::string base = "D:/DX12WithTRT/DataGenerate/";
    m_warp_no_hole = CreateTextureFromExr(base + "warp.exr", DXGI_FORMAT_R32G32B32A32_FLOAT, 1280, 720);
    m_warp_occ = CreateTextureFromExr(base + "warp_occ.exr", DXGI_FORMAT_R32G32B32A32_FLOAT, 1280, 720);
    m_normal = CreateTextureFromExr(base + "normal.exr", DXGI_FORMAT_R32G32B32A32_FLOAT, 1280, 720); 
    
    //m_mask = CreateTextureFromExr("C:/Users/admin/Desktop/TestTRTData/MedievalDocksWrap.0144.exr", DXGI_FORMAT_R32_FLOAT, 1280, 720);
    m_history1 = CreateTextureFromExr(base + "history1.exr", DXGI_FORMAT_R32G32B32A32_FLOAT, 1280, 720);
    m_history2 = CreateTextureFromExr(base + "history2.exr", DXGI_FORMAT_R32G32B32A32_FLOAT, 1280, 720);
    m_history3 = CreateTextureFromExr(base + "history3.exr", DXGI_FORMAT_R32G32B32A32_FLOAT, 1280, 720);
    

    //m_output = CreateDummy(DXGI_FORMAT_R32G32B32A32_FLOAT, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, 1280, 720);
    m_output = CreateTextureFromExr(base + "warp.exr", DXGI_FORMAT_R32G32B32A32_FLOAT, 1280, 720);   //TODO: test compute shader
    m_mask = CreateDummy(DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, 1280, 720);

    CreateSRVandUAV();
}

void DX12CudaInterop::CreateSRVandUAV() {
    m_CbvSrvUavDescriptorSize = m_device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    {
        D3D12_CPU_DESCRIPTOR_HANDLE handle = CD3DX12_CPU_DESCRIPTOR_HANDLE(m_pDescriptorHeap->GetCPUDescriptorHandleForHeapStart(), 0, m_CbvSrvUavDescriptorSize);
        m_device->CreateShaderResourceView(m_history1->defaultBuffer.Get(), nullptr, handle);
    }
    {
        D3D12_CPU_DESCRIPTOR_HANDLE handle = CD3DX12_CPU_DESCRIPTOR_HANDLE(m_pDescriptorHeap->GetCPUDescriptorHandleForHeapStart(), 1, m_CbvSrvUavDescriptorSize);
        m_device->CreateUnorderedAccessView(m_mask->defaultBuffer.Get(), nullptr, nullptr, handle); // TODO: uav  第三个参数 
    }
    {
        D3D12_CPU_DESCRIPTOR_HANDLE handle = CD3DX12_CPU_DESCRIPTOR_HANDLE(m_pDescriptorHeap->GetCPUDescriptorHandleForHeapStart(), 2, m_CbvSrvUavDescriptorSize);
        m_device->CreateShaderResourceView(m_output->defaultBuffer.Get(), nullptr, handle);
    }

    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(m_pDescriptorHeap->GetCPUDescriptorHandleForHeapStart(), 3, m_CbvSrvUavDescriptorSize);

    for (UINT n = 0; n < FrameCount; n++) {
        ThrowIfFailed(
            m_swapChain->GetBuffer(n, IID_PPV_ARGS(&m_renderTargets[n])));
        m_device->CreateUnorderedAccessView(m_renderTargets[n].Get(), nullptr, nullptr,
            rtvHandle);
        rtvHandle.Offset(1, m_CbvSrvUavDescriptorSize);
    }

    
}

void DX12CudaInterop::BindTextureToTRTBindings(std::shared_ptr<Texture> texture, const std::string& inputName) {
    // 0. 创建handle
    HANDLE sharedHandle;
    WindowsSecurityAttributes windowsSecurityAttributes;
    LPCWSTR name = NULL;
    ThrowIfFailed(m_device->CreateSharedHandle(
        texture->defaultBuffer.Get(), &windowsSecurityAttributes, GENERIC_ALL, name,
        &sharedHandle));

    UINT64 requiredSize = GetRequiredIntermediateSize(texture->defaultBuffer.Get(), 0, 1);


    D3D12_RESOURCE_ALLOCATION_INFO d3d12ResourceAllocationInfo;
    d3d12ResourceAllocationInfo = m_device->GetResourceAllocationInfo(
        m_nodeMask, 1, &CD3DX12_RESOURCE_DESC::Buffer(requiredSize));
    size_t actualSize = d3d12ResourceAllocationInfo.SizeInBytes;
    size_t alignment = d3d12ResourceAllocationInfo.Alignment;

    cudaExternalMemoryHandleDesc externalMemoryHandleDesc;
    memset(&externalMemoryHandleDesc, 0, sizeof(externalMemoryHandleDesc));

    externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
    externalMemoryHandleDesc.handle.win32.handle = sharedHandle;
    externalMemoryHandleDesc.size = actualSize;
    externalMemoryHandleDesc.flags = cudaExternalMemoryDedicated;
    // 1. import to cuda
    checkCudaErrors(
        cudaImportExternalMemory(&m_externalMemory, &externalMemoryHandleDesc));

    //CloseHandle(sharedHandle);  //TODO: 需要close ??

    cudaExternalMemoryBufferDesc externalMemoryBufferDesc;
    memset(&externalMemoryBufferDesc, 0, sizeof(externalMemoryBufferDesc));
    externalMemoryBufferDesc.offset = 0;
    externalMemoryBufferDesc.size = requiredSize;
    externalMemoryBufferDesc.flags = 0;


    void* bufferPtr;
    // 2. map到pointer
    checkCudaErrors(cudaExternalMemoryGetMappedBuffer(
        &bufferPtr, m_externalMemory, &externalMemoryBufferDesc));


    m_bufferManager->setBuffer(bufferPtr, inputName);
}
void DX12CudaInterop::ShareTextureWithCuda() {

    BindTextureToTRTBindings(m_history1, "history_1");
    BindTextureToTRTBindings(m_history2, "history_2");
    BindTextureToTRTBindings(m_history3, "history_3");
    BindTextureToTRTBindings(m_warp_no_hole, "warp_no_hole");
    BindTextureToTRTBindings(m_warp_occ, "warp_occ");
    BindTextureToTRTBindings(m_normal, "normal");
    BindTextureToTRTBindings(m_mask, "mask_one_channel");
    BindTextureToTRTBindings(m_output, "output_1");

}


void DX12CudaInterop::BuildComputePipelineAndRS() {
    {
        D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc;
        srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        srvHeapDesc.NodeMask = 0;
        srvHeapDesc.NumDescriptors = 5;
        m_device->CreateDescriptorHeap(
            &srvHeapDesc, IID_PPV_ARGS(&m_pDescriptorHeap));
    }
    {
        CD3DX12_DESCRIPTOR_RANGE maskRange[2];
        CD3DX12_ROOT_PARAMETER parameter[3];

        maskRange[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0);
        maskRange[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0);
        parameter[0].InitAsDescriptorTable(2, maskRange);


        CD3DX12_DESCRIPTOR_RANGE blitRange[1];
        blitRange[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 1);
        parameter[1].InitAsDescriptorTable(1, blitRange);
       
        CD3DX12_DESCRIPTOR_RANGE anotherBlitRange[1];
        anotherBlitRange[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 1);
        parameter[2].InitAsDescriptorTable(1, anotherBlitRange);
        


        CD3DX12_ROOT_SIGNATURE_DESC descRootSignature;
        descRootSignature.Init(3, parameter, 0, nullptr);
        ComPtr<ID3DBlob> pSignature;
        ComPtr<ID3DBlob> pError;
        ThrowIfFailed(D3D12SerializeRootSignature(
            &descRootSignature, D3D_ROOT_SIGNATURE_VERSION_1,
            pSignature.GetAddressOf(), pError.GetAddressOf()));
        ThrowIfFailed(m_device->CreateRootSignature(
            0, pSignature->GetBufferPointer(), pSignature->GetBufferSize(),
            IID_PPV_ARGS(&m_rootSignature)));
    }
    // Create the pipeline state, which includes compiling and loading shaders.
    {
        ComPtr<ID3DBlob> computeShader;

        UINT compileFlags = 0;
        std::wstring filePath = GetAssetFullPath("Blit.hlsl");
        LPCWSTR result = filePath.c_str();
        ThrowIfFailed(D3DCompileFromFile(result, nullptr, nullptr, "main",
            "cs_5_1", compileFlags, 0, &computeShader,
            nullptr));



        // Describe and create the graphics pipeline state object (PSO).
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.CS = CD3DX12_SHADER_BYTECODE(computeShader.Get());
        psoDesc.pRootSignature = m_rootSignature.Get();
        psoDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
        ThrowIfFailed(m_device->CreateComputePipelineState(
            &psoDesc, IID_PPV_ARGS(&m_blitPipeline)));
    }
    {
        ComPtr<ID3DBlob> computeShader;

        UINT compileFlags = 0;
        std::wstring filePath = GetAssetFullPath("Mask.hlsl");
        LPCWSTR result = filePath.c_str();
        ThrowIfFailed(D3DCompileFromFile(result, nullptr, nullptr, "main",
            "cs_5_1", compileFlags, 0, &computeShader,
            nullptr));



        // Describe and create the graphics pipeline state object (PSO).
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.CS = CD3DX12_SHADER_BYTECODE(computeShader.Get());
        psoDesc.pRootSignature = m_rootSignature.Get();
        psoDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
        ThrowIfFailed(m_device->CreateComputePipelineState(
            &psoDesc, IID_PPV_ARGS(&m_maskPipeline)));
    }
}

void DX12CudaInterop::MakeMask() {
    

    m_commandList->SetPipelineState(m_maskPipeline.Get());
    m_commandList->SetComputeRootSignature(m_rootSignature.Get());

    m_commandList->SetComputeRootDescriptorTable(0, CD3DX12_GPU_DESCRIPTOR_HANDLE(m_pDescriptorHeap->GetGPUDescriptorHandleForHeapStart(), 0, m_CbvSrvUavDescriptorSize)); 


    m_commandList->Dispatch(1280, 720, 1);
}
void DX12CudaInterop::Blit() {
    // TODO: 注意Main里保证了swap chain的大小是1280*720 
    m_commandList->SetPipelineState(m_blitPipeline.Get());

    m_commandList->SetComputeRootSignature(m_rootSignature.Get());

    m_commandList->SetComputeRootDescriptorTable(1, CD3DX12_GPU_DESCRIPTOR_HANDLE(m_pDescriptorHeap->GetGPUDescriptorHandleForHeapStart(), 2, m_CbvSrvUavDescriptorSize)); 
    m_commandList->SetComputeRootDescriptorTable(2, CD3DX12_GPU_DESCRIPTOR_HANDLE(m_pDescriptorHeap->GetGPUDescriptorHandleForHeapStart(), 3+m_frameIndex, m_CbvSrvUavDescriptorSize)); 

    
    m_commandList->Dispatch(1280, 720, 1);
}


void DX12CudaInterop::DXSignalAndCudaWait() {
    /*ThrowIfFailed(m_commandList->Close());
    ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
    m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

    const UINT64 currentFenceValue = m_fenceValues[m_frameIndex];
    ThrowIfFailed(m_commandQueue->Signal(m_fence.Get(), currentFenceValue));

    const UINT64 currentFenceValue = m_fenceValues[m_frameIndex];
    cudaExternalSemaphoreWaitParams externalSemaphoreWaitParams;
    memset(&externalSemaphoreWaitParams, 0, sizeof(externalSemaphoreWaitParams));

    externalSemaphoreWaitParams.params.fence.value = currentFenceValue;
    externalSemaphoreWaitParams.flags = 0;

    checkCudaErrors(cudaWaitExternalSemaphoresAsync(
        &m_externalSemaphore, &externalSemaphoreWaitParams, 1, m_streamToRun));*/

}

void DX12CudaInterop::CudaSignalAndDXWait() {
    //cudaExternalSemaphoreSignalParams externalSemaphoreSignalParams;
    //memset(&externalSemaphoreSignalParams, 0,
    //    sizeof(externalSemaphoreSignalParams));
    //m_fenceValues[m_frameIndex] = currentFenceValue + 1;
    //externalSemaphoreSignalParams.params.fence.value =
    //    m_fenceValues[m_frameIndex];
    //externalSemaphoreSignalParams.flags = 0;

    //checkCudaErrors(cudaSignalExternalSemaphoresAsync(
    //    &m_externalSemaphore, &externalSemaphoreSignalParams, 1, m_streamToRun));

    //// Update the frame index.
    //m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

    //// If the next frame is not ready to be rendered yet, wait until it is ready.
    //if (m_fence->GetCompletedValue() < m_fenceValues[m_frameIndex]) {
    //    ThrowIfFailed(m_fence->SetEventOnCompletion(m_fenceValues[m_frameIndex],
    //        m_fenceEvent));
    //    WaitForSingleObjectEx(m_fenceEvent, INFINITE, FALSE);
    //}

    //// Set the fence value for the next frame.
    //m_fenceValues[m_frameIndex] = currentFenceValue + 2;
}

// Render the scene.
void DX12CudaInterop::OnRender() {

    ThrowIfFailed(m_commandAllocators[m_frameIndex]->Reset());
    ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(),
        m_maskPipeline.Get()));


    ID3D12DescriptorHeap* ppHeaps[] = { m_pDescriptorHeap.Get() };
    m_commandList->SetDescriptorHeaps(1, ppHeaps);
    MakeMask();
     

    //DXSignalAndCudaWait();
    //bool success = m_onnxInferEngine->infer(m_bufferManager->GetRawData());
    //CudaSignalAndDXWait(); // TODO: 是否必要？ 
    /*ThrowIfFailed(m_commandAllocators[m_frameIndex]->Reset());
    ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(),
        m_maskPipeline.Get()));*/
    /*ID3D12DescriptorHeap* ppHeaps[] = { m_pDescriptorHeap.Get() };
    m_commandList->SetDescriptorHeaps(1, ppHeaps);*/

    m_commandList->ResourceBarrier(
        1, &CD3DX12_RESOURCE_BARRIER::Transition(
            m_renderTargets[m_frameIndex].Get(),
            D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
    Blit();
    m_commandList->ResourceBarrier(
        1, &CD3DX12_RESOURCE_BARRIER::Transition(
            m_renderTargets[m_frameIndex].Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PRESENT));



    ThrowIfFailed(m_commandList->Close());

    // Execute the command list.
    ID3D12CommandList *ppCommandLists[] = {m_commandList.Get()};
    m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

    // Present the frame.
    ThrowIfFailed(m_swapChain->Present(1, 0));

    //// Schedule a Signal command in the queue.
    //const UINT64 currentFenceValue = m_fenceValues[m_frameIndex];
    //ThrowIfFailed(m_commandQueue->Signal(m_fence.Get(), currentFenceValue));


    // Update the frame index.
    m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

    //// If the next frame is not ready to be rendered yet, wait until it is ready.
    //if (m_fence->GetCompletedValue() < m_fenceValues[m_frameIndex])
    //{
    //    ThrowIfFailed(m_fence->SetEventOnCompletion(m_fenceValues[m_frameIndex], m_fenceEvent));
    //    WaitForSingleObjectEx(m_fenceEvent, INFINITE, FALSE);
    //}

    //// Set the fence value for the next frame.
    //m_fenceValues[m_frameIndex] = currentFenceValue + 1;


    //MoveToNextFrame();
}

void DX12CudaInterop::OnDestroy() {
  // Ensure that the GPU is no longer referencing resources that are about to be
  // cleaned up by the destructor.
  WaitForGpu();
  /*checkCudaErrors(cudaDestroyExternalSemaphore(m_externalSemaphore));
  checkCudaErrors(cudaDestroyExternalMemory(m_externalMemory));*/
  CloseHandle(m_fenceEvent);
}


// Wait for pending GPU work to complete.
void DX12CudaInterop::WaitForGpu() {
  // Schedule a Signal command in the queue.
  ThrowIfFailed(
      m_commandQueue->Signal(m_fence.Get(), m_fenceValues[m_frameIndex]));

  // Wait until the fence has been processed.
  ThrowIfFailed(
      m_fence->SetEventOnCompletion(m_fenceValues[m_frameIndex], m_fenceEvent));
  WaitForSingleObjectEx(m_fenceEvent, INFINITE, FALSE);

  // Increment the fence value for the current frame.
  m_fenceValues[m_frameIndex]++;
}

// Prepare to render the next frame.
// TODO: 这里包含两个部分  
// 1. dx的command queue signal, cuda 等待
// 2. cuda执行后signal  dx等待
void DX12CudaInterop::MoveToNextFrame() {
  const UINT64 currentFenceValue = m_fenceValues[m_frameIndex];
  cudaExternalSemaphoreWaitParams externalSemaphoreWaitParams;
  memset(&externalSemaphoreWaitParams, 0, sizeof(externalSemaphoreWaitParams));

  externalSemaphoreWaitParams.params.fence.value = currentFenceValue;
  externalSemaphoreWaitParams.flags = 0;

  checkCudaErrors(cudaWaitExternalSemaphoresAsync(
      &m_externalSemaphore, &externalSemaphoreWaitParams, 1, m_streamToRun));

  m_AnimTime += 0.01f;
  RunSineWaveKernel(vertBufWidth, vertBufHeight, (Vertex *)m_cudaDevVertptr,
                    m_streamToRun, m_AnimTime);

  cudaExternalSemaphoreSignalParams externalSemaphoreSignalParams;
  memset(&externalSemaphoreSignalParams, 0,
         sizeof(externalSemaphoreSignalParams));
  m_fenceValues[m_frameIndex] = currentFenceValue + 1;
  externalSemaphoreSignalParams.params.fence.value =
      m_fenceValues[m_frameIndex];
  externalSemaphoreSignalParams.flags = 0;

  checkCudaErrors(cudaSignalExternalSemaphoresAsync(
      &m_externalSemaphore, &externalSemaphoreSignalParams, 1, m_streamToRun));

  // Update the frame index.
  m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

  // If the next frame is not ready to be rendered yet, wait until it is ready.
  if (m_fence->GetCompletedValue() < m_fenceValues[m_frameIndex]) {
    ThrowIfFailed(m_fence->SetEventOnCompletion(m_fenceValues[m_frameIndex],
                                                m_fenceEvent));
    WaitForSingleObjectEx(m_fenceEvent, INFINITE, FALSE);
  }

  // Set the fence value for the next frame.
  m_fenceValues[m_frameIndex] = currentFenceValue + 2;
}
