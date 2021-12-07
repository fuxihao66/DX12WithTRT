# 准备dx12 trt测试需要的数据
import cv2
import OpenEXR
import Imath
import numpy as np
basepath = r"D:\ExtraNet\TestData/"
warp_occ = cv2.imread(basepath+"occ/MedievalDocksWarp.0339.1.exr", cv2.IMREAD_UNCHANGED)
warp = cv2.imread(basepath+"warp_no_hole/MedievalDocksWarp.0339.1.exr", cv2.IMREAD_UNCHANGED)
normal = cv2.imread(basepath+"MedievalDocksWorldNormal.0339.exr", cv2.IMREAD_UNCHANGED)
depth = cv2.imread(basepath+"MedievalDocksSceneDepth.0339.exr", cv2.IMREAD_UNCHANGED)[:,:,0:1]

depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))

roughness = cv2.imread(basepath+"MedievalDocksRoughness.0339.exr", cv2.IMREAD_UNCHANGED)[:,:,0:1]
metallic = cv2.imread(basepath+"MedievalDocksMetallic.0339.exr", cv2.IMREAD_UNCHANGED)[:,:,0:1]

history1 = warp.copy()
history2 = cv2.imread(basepath+"warp_no_hole/MedievalDocksWarp.0339.3.exr", cv2.IMREAD_UNCHANGED)
history3 = cv2.imread(basepath+"warp_no_hole/MedievalDocksWarp.0339.5.exr", cv2.IMREAD_UNCHANGED)

mask1 = cv2.imread(basepath+"warp_res/MedievalDocksWarp.0339.1.exr", cv2.IMREAD_UNCHANGED)[:,:,0:1]
mask2 = cv2.imread(basepath+"warp_res/MedievalDocksWarp.0339.3.exr", cv2.IMREAD_UNCHANGED)[:,:,0:1]
mask3 = cv2.imread(basepath+"warp_res/MedievalDocksWarp.0339.5.exr", cv2.IMREAD_UNCHANGED)[:,:,0:1]
mask1[mask1 >= 0] = 1.
mask1[mask1 < 0] = 0.
mask2[mask2 >= 0] = 1.
mask2[mask2 < 0] = 0.
mask3[mask3 >= 0] = 1.
mask3[mask3 < 0] = 0.
warp[warp < 0] = 0.
warp_occ[warp_occ < 0] = 0.
history1[history1 < 0] = 0.
history2[history2 < 0] = 0.
history3[history3 < 0] = 0.



def saveImage(img, alpha, path):
    
    newImg = img
    newAlpha = alpha
    header = OpenEXR.Header(1280,720)
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header['channels'] = dict([(c, half_chan) for c in "RGBA"])
    exr = OpenEXR.OutputFile(path, header)
    
    # exr.writePixels({'R': newImg[:,:,2].tobytes(), 'G': newImg[:,:,1].tobytes(), 'B': newImg[:,:,0].tobytes()})
    exr.writePixels({'R': newImg[:,:,2].tobytes(), 'G': newImg[:,:,1].tobytes(), 'B': newImg[:,:,0].tobytes(), 'A': newAlpha[:,:,0].tobytes()})
    exr.close()


saveImage(history1, mask1, "history1.exr")
saveImage(history2, mask2, "history2.exr")
saveImage(history3, mask3, "history3.exr")
saveImage(warp, roughness, "warp.exr")
saveImage(warp_occ, metallic, "warp_occ.exr")
saveImage(normal, depth, "normal.exr")

