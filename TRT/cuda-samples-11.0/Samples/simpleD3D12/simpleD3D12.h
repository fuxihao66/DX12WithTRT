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

#pragma once

#include "DX12CudaSample.h"
#include "ShaderStructs.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"
#include "NvOnnxParser.h"
#include "logger.h"
#include "logging.h"
#include "argsParser.h"
#include "common.h"
#include <vector>
using namespace DirectX;

// Note that while ComPtr is used to manage the lifetime of resources on the
// CPU, it has no understanding of the lifetime of resources on the GPU. Apps
// must account for the GPU lifetime of resources to avoid destroying objects
// that may still be referenced by the GPU. An example of this can be found in
// the class method: OnDestroy().
using Microsoft::WRL::ComPtr;
class Texture {
public:
    ComPtr<ID3D12Resource> defaultBuffer;
    ComPtr<ID3D12Resource> uploadBuffer;
};


class OnnxEngine
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    OnnxEngine(const samplesCommon::OnnxSampleParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
    }
    bool infer(void** rawBufferData) {
        auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
        if (!context)
        {
            return false;
        }
        bool status = context->executeV2(rawBufferData);
        if (!status)
        {
            return false;
        }

        return true;
    }
    bool build() {
        auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
        if (!builder)
        {
            return false;
        }

        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
        if (!network)
        {
            return false;
        }
        auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config)
        {
            return false;
        }

        auto parser
            = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
        if (!parser)
        {
            return false;
        }

        auto constructed = constructNetwork(builder, network, config, parser);
        if (!constructed)
        {
            return false;
        }
        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
        if (!mEngine)
        {
            return false;
        }
        /*sample::gLogInfo << network->getInput(0)->getAllowedFormats() << endl;
        sample::gLogInfo << "Num of inputs: " << network->getNbInputs() << endl;*/
        mInputDims = network->getInput(0)->getDimensions();
        assert(mInputDims.nbDims == 4);

        //assert(network->getNbOutputs() == 1);
        mOutputDims = network->getOutput(0)->getDimensions();
        //assert(mOutputDims.nbDims == 2);
    }
    std::shared_ptr<nvinfer1::ICudaEngine> GetEngine() {
        return mEngine;
    }
private:
    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    int mNumber{ 0 };             //!< The number to classify

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    
    
    
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser) {
        auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
            static_cast<int>(sample::gLogger.getReportableSeverity()));
        if (!parsed)
        {
            return false;
        }

        config->setMaxWorkspaceSize(512_MiB);
        
        config->setFlag(BuilderFlag::kFP16);
        
        samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

        return true;
    }

};

class NetworkBufferManager {
    std::vector<void*> _buffers;
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
public:
    NetworkBufferManager() = default;
    NetworkBufferManager(std::shared_ptr<nvinfer1::ICudaEngine> _pEngine) {
        mEngine = _pEngine;
        _resize(mEngine->getNbBindings());
    }
    
    void* getBuffer(const std::string& tensorName) const
    {
        int index = mEngine->getBindingIndex(tensorName.c_str());
        if (index == -1)
            return nullptr;
        return _buffers[index];
    }
    bool setBuffer(void* dataPtr, const std::string& tensorName) {
        int index = mEngine->getBindingIndex(tensorName.c_str());
        if (index == -1)
            return false;
        _buffers[index] = dataPtr;
        return true; 
    }
    void** GetRawData() {
        return _buffers.data();
    }
private:
    void _resize(unsigned int numBindings) {
        _buffers.resize(numBindings);
    }
};


static const char *shaderstr =
    " struct PSInput \n"
    " { \n"
    "  float4 position : SV_POSITION; \n"
    "  float4 color : COLOR; \n"
    " } \n"
    " PSInput VSMain(float3 position : POSITION, float4 color : COLOR) \n"
    " { \n"
    "  PSInput result;\n"
    "  result.position = float4(position, 1.0f);\n"
    "  result.color = color;\n"
    "  return result; \n"
    " } \n"
    " float4 PSMain(PSInput input) : SV_TARGET \n"
    " { \n"
    "  return input.color;\n"
    " } \n";

class DX12CudaInterop : public DX12CudaSample {
 public:
  DX12CudaInterop(UINT width, UINT height, std::string name);

  virtual void OnInit();
  virtual void OnRender();
  virtual void OnDestroy();

 private:
  // In this sample we overload the meaning of FrameCount to mean both the
  // maximum number of frames that will be queued to the GPU at a time, as well
  // as the number of back buffers in the DXGI swap chain. For the majority of
  // applications, this is convenient and works well. However, there will be
  // certain cases where an application may want to queue up more frames than
  // there are back buffers available. It should be noted that excessive
  // buffering of frames dependent on user input may result in noticeable
  // latency in your app.
  static const UINT FrameCount = 2;
  std::string shadersSrc = shaderstr;
#if 0
		" struct PSInput \n" \
		" { \n" \
		"  float4 position : SV_POSITION; \n" \
		"  float4 color : COLOR; \n" \
		" } \n" \
		" PSInput VSMain(float3 position : POSITION, float4 color : COLOR) \n" \
		" { \n" \
		"  PSInput result;\n" \
		"  result.position = float4(position, 1.0f);\n" \
		"  result.color = color;\n"	\
		"  return result; \n" \
		" } \n" \
		" float4 PSMain(PSInput input) : SV_TARGET \n" \
		" { \n" \
		"  return input.color;\n" \
		" } \n";
#endif

  // Vertex Buffer dimension
  size_t vertBufHeight, vertBufWidth;

  // Pipeline objects.
  D3D12_VIEWPORT m_viewport;
  CD3DX12_RECT m_scissorRect;
  ComPtr<IDXGISwapChain3> m_swapChain;
  ComPtr<ID3D12Device> m_device;
  ComPtr<ID3D12Resource> m_renderTargets[FrameCount];
  ComPtr<ID3D12CommandAllocator> m_commandAllocators[FrameCount];
  ComPtr<ID3D12CommandQueue> m_commandQueue;
  ComPtr<ID3D12RootSignature> m_rootSignature;
  ComPtr<ID3D12DescriptorHeap> m_rtvHeap;
  ComPtr<ID3D12PipelineState> m_pipelineState;
  ComPtr<ID3D12GraphicsCommandList> m_commandList;
  //UINT m_rtvDescriptorSize;


  // TODO: Custom Definition 
  UINT m_CbvSrvUavDescriptorSize;
  std::shared_ptr<Texture> m_warp_no_hole;
  std::shared_ptr<Texture> m_warp_occ;
  std::shared_ptr<Texture> m_normal;
  std::shared_ptr<Texture> m_mask;
  std::shared_ptr<Texture> m_history1;
  std::shared_ptr<Texture> m_history2;
  std::shared_ptr<Texture> m_history3;
  std::shared_ptr<Texture> m_output;

  //CD3DX12_GPU_DESCRIPTOR_HANDLE m_output_rtv;

  void PrepareResources();
  std::shared_ptr<Texture> CreateTextureFromExr(const std::string& filePath, DXGI_FORMAT textureFormat, unsigned int width, unsigned int height);
  void Blit();
  void MakeMask();
  std::shared_ptr<Texture> CreateDummy(DXGI_FORMAT textureFormat, D3D12_RESOURCE_STATES initState, unsigned int width, unsigned int height);
  bool LoadONNXModel(const std::string& filePath);
  void BindTextureToTRTBindings(std::shared_ptr<Texture> texture, const std::string& inputName);
  void ShareTextureWithCuda();
  void BuildComputePipelineAndRS();
  void CreateSRVandUAV();

  void DXSignalAndCudaWait();
  void CudaSignalAndDXWait();

  std::shared_ptr<OnnxEngine> m_onnxInferEngine;

  ComPtr<ID3D12PipelineState> m_maskPipeline;
  ComPtr<ID3D12PipelineState> m_blitPipeline;
  
  ComPtr<ID3D12DescriptorHeap> m_pDescriptorHeap;

  std::shared_ptr<NetworkBufferManager> m_bufferManager;
  // TODO: End Custom Definition 

  // App resources.
  ComPtr<ID3D12Resource> m_vertexBuffer;
  D3D12_VERTEX_BUFFER_VIEW m_vertexBufferView;

  // Synchronization objects.
  UINT m_frameIndex;
  HANDLE m_fenceEvent;
  ComPtr<ID3D12Fence> m_fence;
  UINT64 m_fenceValues[FrameCount];

  // CUDA objects
  cudaExternalMemoryHandleType m_externalMemoryHandleType;
  cudaExternalMemory_t m_externalMemory;
  cudaExternalSemaphore_t m_externalSemaphore;
  cudaStream_t m_streamToRun;
  LUID m_dx12deviceluid;
  UINT m_cudaDeviceID;
  UINT m_nodeMask;
  float m_AnimTime;
  void *m_cudaDevVertptr = NULL;

  void LoadPipeline();
  void InitCuda();
  void LoadAssets();
  //void PopulateCommandList();
  void MoveToNextFrame();
  void WaitForGpu();
};
