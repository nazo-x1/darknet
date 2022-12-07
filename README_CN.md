后端构建说明：

1. 大环境依赖：

   CUDA toolkit

   cuDNN

   OpenCV

   python3

2. clone 或者下载本项目 

   ```shell
   git clone https://ghproxy.com/https://github.com/nazo-x1/darknet.git
   ```

   

2. 首先需要编译主程序与库文件（不同操作系统）

   1. Linux：

      目前未尝试

      参照英文 README  `How to compile on Linux/macOS (using CMake)` 条目

      make 前 按需选好 GPU 等项即可

   2. win：

      注意 CUDA CUDNN 等要在环境变量，OPENCV 好像不需要特别安装

      在项目文件夹下

      使用 powershell 运行 build.ps1 脚本
      
      ```powershell
      ./build.ps1 -UseVCPKG -EnableCUDA -EnableCUDNN # -EnableOPENCV
      ```

      顺利的话很快，注意及时解决报错 可以尝试 去掉 GPU 相关的 -Enable
      
      没有训练模型和绘制训练进程的需求可以选择不开 OPENCV
      
      -UseVCPKG 会使用 VCPKG 包管理，帮助解决一些依赖问题

3. 编译完成后会生成若干文件，其中 darknet.exe 为 C 语言主程序（官方demo），

   本项目运行 main.py，在 python 下使用编译生成的 C 的库函数

   （最好安装使用官网的 python3.7，否则python 调用 C 语言库会出现问题

   部署具体过程

   1. ```shell
      pip install -r requirements.txt
      ```

   2. ```shell
      python main.py --mode flask
      ```

   3. 运行后如果缺失 zlibwapi.dll，
   
      参考 [Installation Guide :: NVIDIA Deep Learning cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-zlib-windows)
   
      
