后端构建说明：

1. 大环境依赖：

   CUDA toolkit

   cuDNN

   OpenCV

   python3

2. 首先需要编译主程序与库文件（不同操作系统）

   1. Linux：

      目前未尝试

      参照英文 README  `How to compile on Linux/macOS (using CMake)` 条目

      make 前 按需选好 GPU 等项即可

   2. win：

      注意 CUDA CUDNN 等要在环境变量，OPENCV 好像不需要特别安装

      使用 powershell 运行 build.ps1 脚本

      ```powershell
      ./build.ps1 -UseVCPKG -EnableOPENCV -EnableCUDA -EnableCUDNN
      ```

      耗时较久，注意及时解决报错

      

3. 编译完成后会生成若干文件，其中 darknet.exe 为 C 语言主程序（官方demo），

   本项目运行 main.py，在 python 下使用 C 的库函数

   最好使用官网的 python3.7 安装 python，否则python 调用 C 语言库会出现问题

   具体过程

   1. ```shell
      pip install -r requirements.txt
      ```

   2. ```
      python main.py --mode flask
      ```

      


​      



