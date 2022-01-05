

Texture2D<float4> gNetworkOutput : register(t0, space1); 
RWTexture2D<float4> gRenderTarget : register(u0, space1);

[numthreads(1, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    int2 PixelPos = DTid.xy;
    
    gRenderTarget[PixelPos] = float4(gNetworkOutput[PixelPos].xyz, 1.0f);

}

