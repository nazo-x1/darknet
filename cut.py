from pathlib import Path
import cv2
import os

def _split(or_img: Path, sp_imgdir: str = "", targetHeight: int = 512, targetWidth: int = 512, overlap: int = 2) -> None:
    '分割指定路径文件, 保存到指定目录'
    oriimg = cv2.imread(f'{or_img.__str__()}')
    print(f"cutting {or_img.__str__()}")
    image_copy = oriimg.copy()
    oriHeight = oriimg.shape[0]
    oriwidth = oriimg.shape[1]

    assert(targetHeight % overlap == 0)
    assert(targetWidth % overlap == 0)

    for y in range(0, oriHeight, targetHeight//overlap):
        for x in range(0, oriwidth, targetWidth//overlap):
            y1 = y + targetHeight
            x1 = x + targetWidth

            # check whether the patch width or height exceeds the image width or height
            if x1 >= oriwidth and y1 >= oriHeight:
                x1 = oriwidth - 1
                y1 = oriHeight - 1
            elif y1 >= oriHeight:  # when patch height exceeds the image height
                y1 = oriHeight - 1
            elif x1 >= oriwidth:  # when patch width exceeds the image width
                x1 = oriwidth - 1

            x = x1-targetWidth
            y = y1-targetHeight

            tiles = image_copy[y:y1, x:x1]
            # print(f'{sp_imgdir}{x}_{y}_{or_img.name}')
            cv2.imwrite(
                f'{sp_imgdir}{x}_{y}_{or_img.name}', tiles)

def _split_2(or_img: Path, targetHeight: int = 512, targetWidth: int = 512, overlap: int = 2):
    '分割指定路径文件, 返回图片列表'
    RES = {}
    oriimg = cv2.imread(f'{or_img.__str__()}')
    print(f"cutting {or_img.__str__()}")
    image_copy = oriimg.copy()
    oriHeight = oriimg.shape[0]
    oriwidth = oriimg.shape[1]

    assert(targetHeight % overlap == 0)
    assert(targetWidth % overlap == 0)

    for y in range(0, oriHeight, targetHeight//overlap):
        for x in range(0, oriwidth, targetWidth//overlap):
            y1 = y + targetHeight
            x1 = x + targetWidth

            # check whether the patch width or height exceeds the image width or height
            if x1 >= oriwidth and y1 >= oriHeight:
                x1 = oriwidth - 1
                y1 = oriHeight - 1
            elif y1 >= oriHeight:  # when patch height exceeds the image height
                y1 = oriHeight - 1
            elif x1 >= oriwidth:  # when patch width exceeds the image width
                x1 = oriwidth - 1

            x = x1-targetWidth
            y = y1-targetHeight

            tiles = image_copy[y:y1, x:x1]
            # print(f'{sp_imgdir}{x}_{y}_{or_img.name}')
            # cv2.imwrite(
            #     f'{sp_imgdir}{x}_{y}_{or_img.name}', tiles)
            RES[f"{x}_{y}_{or_img.name}"] = tiles.copy()
    return RES

def cutimg(fname:Path):
    '分割指定图片, 返回裁剪后的目录'
    if(not fname.exists()):
        print("file not exist")
        return "{}"
    if(not fname.is_file()):
        print("not a img")
        return "{}"
    imgpath = fname.name.rsplit(".",1)[0]
    if(not os.path.exists(imgpath)):
        os.mkdir(imgpath)
    _split(fname, imgpath+"/", 512, 512, 2)
    return imgpath
