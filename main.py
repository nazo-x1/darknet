from io import BytesIO
import json
import os
from pathlib import Path
import random
import time
from cut import _split_2
import darknet
import argparse
from flask import Flask,request,render_template
from PIL import Image,ImageDraw
import cv2
import base64

app = Flask(__name__,static_url_path="")

def parser():
    parser = argparse.ArgumentParser(description="焊点检测")
    parser.add_argument("--mode", type=str, default="console",
                        help="模式选择"
                        "\nconsole 命令行模式 从命令行获取图片路径 img"
                        "\n;flask 后端模式",required=True)
    parser.add_argument("--img", type=str, default="",
                        help="图片路径 命令行模式下传入 相对绝对均可")
    parser.add_argument("--batch_size", default=4, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--weights", default="backup\pos_final.weights",
                        help="权重矩阵")
    parser.add_argument("--config_file", default="./cfg/pos.cfg",
                        help="配置文件")
    parser.add_argument("--data_file", default="./cfg/pos.data",
                        help="数据描述文件")

    parser.add_argument("--thresh", type=float, default=.8,
                        help="remove detections with lower confidence")
    return parser.parse_args()

def genIMG(img:Path):
    # 图片切割 -- 优化中
    if(not img.exists()):
        print("file not exist")
        exit("file not exist")
    if(not img.is_file()):
        print("not a img")
        exit("not a img")
    prev_time = time.time()

    RES = _split_2(img)
    cost = (time.time() - prev_time)
    print("cutting cost: {}s".format(cost))
    return RES

def getPOS(images:dict,network,class_names,class_colors):
    # 小图坐标识别 -- 待优化
    POSSET = {}
    prev_time = time.time()

    width = darknet.network_width(network)
    height = darknet.network_height(network)

    darknet_image_BUFF = darknet.make_image(width, height, 3)

    for image_name,im in images.items():
        # 图片预处理
        # assert type(im) == numpy.ndarray
        image_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),interpolation=cv2.INTER_LINEAR)
        # 移入缓冲区
        darknet.copy_image_from_bytes(darknet_image_BUFF, image_resized.tobytes())
        # 识别
        detections = darknet.detect_image(network, class_names, darknet_image_BUFF, thresh=.8)
        POSSET[image_name]=detections
    darknet.free_image(darknet_image_BUFF)

    cost = (time.time() - prev_time)
    print("detect cost: {}s".format(cost))
    return POSSET

def genRealpos(POSSET):
    # 合成大图坐标
    prev_time = time.time()
    realpos = {}
    count = 0
    for name, posList in POSSET.items(): # 遍历每张小图
        x_offset,y_offset,_ = Path(name).name.split("_")
        x_offset,y_offset = int(x_offset),int(y_offset)
        for pos in posList: # 遍历每个识别到的焊点
            POSA = {}
            xAc,yAc,width_A,length_A = pos[2]
            POSA['xmin'] = x_offset + xAc - width_A/2
            POSA['xmax'] = x_offset + xAc + width_A/2
            POSA['ymin'] = y_offset + yAc - length_A/2
            POSA['ymax'] = y_offset + yAc + length_A/2
            POSA['class'] = pos[0]
            POSA['conf'] = float(pos[1])

            xiangjiao_flag = False
            for id, POSB in realpos.items(): # 相交矩形处理
                # 计算两个点中心
                xAcenter_2x = POSA['xmax'] + POSA['xmin'] # / 2
                # print(xAc,xmiddleA_double) 注意 xAc 还是相对地址
                xBcenter_2x = POSB['xmax'] + POSB['xmin'] # / 2
                yAcenter_2x = POSA['ymax'] + POSA['ymin'] # / 2
                yBcenter_2x = POSB['ymax'] + POSB['ymin'] # / 2
                # 计算两个矩形长宽
                width_A = POSA['xmax'] - POSA['xmin']
                width_B = POSB['xmax'] - POSB['xmin']
                length_A = POSA['ymax'] - POSA['ymin']
                length_B = POSB['ymax'] - POSB['ymin']
                # 比较中心距和长宽和, 下面判断 +1 以放宽相交的条件
                # if(abs(xAcenter_2x-xBcenter_2x) <= abs(width_A + width_B) and abs(yAcenter_2x-yBcenter_2x) <= abs(length_A + length_B)):
                # 当一个矩形在另一个中心时进行合并
                if(xAcenter_2x <= POSB['xmax']*2 and xAcenter_2x >= POSB['xmin']*2 and yAcenter_2x <= POSB['ymax']*2 and yAcenter_2x >= POSB['ymin']*2) or \
                  (xBcenter_2x <= POSA['xmax']*2 and xBcenter_2x >= POSA['xmin']*2 and yBcenter_2x <= POSA['ymax']*2 and yBcenter_2x >= POSA['ymin']*2):
                    # print("存在相交！",POSA,POSB)
                    xiangjiao_flag = True
                    POSA['xmin'] = min(POSA['xmin'],POSB['xmin'])
                    POSA['ymin'] = min(POSA['ymin'],POSB['ymin'])
                    POSA['xmax'] = max(POSA['xmax'],POSB['xmax'])
                    POSA['ymax'] = max(POSA['ymax'],POSB['ymax'])
                    POSA['class'] = POSA['class'] + (POSB['class'] if not (POSB['class'] in POSA['class']) else "")
                    POSA['conf'] = max(POSA['conf'],POSB['conf'])
                    realpos[id] = POSA
            if(xiangjiao_flag == False):
                realpos[count] = POSA
                count+=1

    cost = (time.time() - prev_time)
    print("genpos cost: {}s".format(cost))
    # print(realpos)
    return realpos

def console_main(args):
    img = Path(args.img)
    images = genIMG(img) # 不一定要把图片切到磁盘
    POSSET = getPOS(images,network,class_names,class_colors)
    realpos = genRealpos(POSSET)
    s = json.dumps(realpos,indent=3)
    with open("jsons/"+img.name+".json","w") as f: f.write(s)
    return

@app.route('/',methods=["GET",'POST'])
def index():
    if request.method == "GET":
        return render_template('index.html')
    if request.method == "POST":
        print(request.data)
        file = request.files['image']
        filename = file.filename
        file.save(filename)
        img = Path(filename)
        images = genIMG(img) # 不一定要把图片切到磁盘
        POSSET = getPOS(images,network,class_names,class_colors)
        realpos = genRealpos(POSSET)

        # 绘图
        im = Image.open(img).copy()
        drawer=ImageDraw.ImageDraw(im)
        for _,pos in realpos.items():
            drawer.rectangle((pos['xmin'],pos['ymin'],pos['xmax'],pos['ymax']),outline ='red',width = 5)
        output_buffer = BytesIO()
        im.save(output_buffer, quality=100, format="JPEG")
        byte_data = output_buffer.getvalue()
        base64_str = 'data:image/' + 'jpeg' + ';base64,' + base64.b64encode(byte_data).decode("utf-8")

        context = {}
        context["b64img"] = base64_str
        context["realpos"] = realpos
        os.remove(img)
        return render_template("index2.html",**context)

@app.route('/imgpos',methods=["GET",'POST'])
def upload():
    if request.method == "GET":
        return render_template('upload.html')
    if request.method == "POST":
        file = request.files['image']
        filename = file.filename
        file.save(filename)
        img = Path(filename)
        images = genIMG(img) # 不一定要把图片切到磁盘
        POSSET = getPOS(images,network,class_names,class_colors)
        realpos = genRealpos(POSSET)

        # 绘图
        im = Image.open(img).copy()
        drawer=ImageDraw.ImageDraw(im)
        for _,pos in realpos.items():
            drawer.rectangle((pos['xmin'],pos['ymin'],pos['xmax'],pos['ymax']),outline ='red',width = 5)
        # if(len(realpos) != 52):
        #     print(img.name)
        #     im.show()

        os.remove(filename)
        return realpos

def flask_main(args):
    app.run(debug=True,host="0.0.0.0")
    return

if __name__ == "__main__":
    args = parser()

    # 模型预加载
    batch_size = args.batch_size
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=batch_size
    )

    if(args.mode == "console"):
        console_main(args)
    elif(args.mode == "flask"):
        flask_main(args)
    else:
        exit("unknow mode")
