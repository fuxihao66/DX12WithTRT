
Texture2D<float4> gHistory1 : register(t0, space0); 
RWTexture2D<float> gMask : register(u0, space0);

[numthreads(1, 1, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
    int2 PixelPos = DTid.xy;
    
    gMask[PixelPos].x = gHistory1[PixelPos].w;

}
