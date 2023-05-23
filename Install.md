This fork aims to reproduce the results mentioned in the paper in 2023 from the pespective of a software developer


Developer environment:
* Windows 10/11
* Visual Studio 2022 17.5.1
* anaconda(any version should be ok)
* cuda SDK 11.8

If you want to have numerical identical inference results of the original paper, please follow the step here:
The repository is inherited from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark, which is completely outdated, and stopped to update since 4 years ago. Therefore we need to first get the closest pytorch installation that will make maskrcnn-benchmark build.

1. conda create -n pytorch-1.0
2. conda activate pytorch-1.0
2. conda install python=3.7
4. install pytorch 1.0 with whl  
    <code>pip install https://download.pytorch.org/whl/cu100/torch-1.0.0-cp37-cp37m-win_amd64.whl</code>
    verify that the installation is sucessful
    <code>python -c "import torch;print(torch.__version__)"</code>
5. install torchvision
    <code>pip install https://download.pytorch.org/whl/torchvision-0.2.0-py2.py3-none-any.whl</code>
6. now start to build the cpp and cu files
   1. go to source code root dir
   2.  <code>python -c "import torch;from torch.utils.cpp_extension import CUDA_HOME;print (CUDA_HOME)"</code> it is going to look for CUDA_PATH environment variable in the system, if it is not found, it outputs <code>C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA\v10.2</code>.  pytorch 1.0 is likely built with cuda 10.2. But in my system it shows <code>C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA\v11.8</code>
   3. run <code>"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"</code> so that cl.exe can be found in the path
   4. build the cpp and cu extensions by running <code>python setup.py build_ext</code>,and 
7. install this module,<code>python setup.py install</code>
8. (hack)Now there should be an new folder <code>build</code> created in the root dir. For some reason the <code>align</code> folder under <code>$(STR-TDSL_RootDir)/maskrcnn_benchmark/modeling/one_stage_head</code> is not copied. Therefore on my computer i have to do the following manual copying to <code>$(AnacondaRootDir)\envs\pytorch-1.0\Lib\site-packages\maskrcnn_benchmark-0.1-py3.7-win-amd64.egg\maskrcnn_benchmark\modeling\one_stage_head</code>

If you would like to work with the latst pytorch,please switch to pytorch-2.x branch of this repository.
All the cuda code in [maskrcnn-benchmark] are patched to be compatible with pytorch-2.x. The patches will likely to work with any reasonable not-so-outdated pytorch.
1. install latest pytorch following the guide in pytorch official website
2. now start to build the cpp and cu files
    1. go to source code root dir
    2.<code>python -c "import torch;from torch.utils.cpp_extension import CUDA_HOME;print (CUDA_HOME)"</code> it is going to look for CUDA_PATH environment variable in the system, if it is not found, it outputs <code>C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA\v10.2</code>.  pytorch 1.0 is likely built with cuda 10.2. But in my system it shows <code>C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA\v11.8</code>
    3. run <code>"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"</code> so that cl.exe can be found in the path
    4. build the cpp and cu extensions by running <code>python setup.py build_ext</code>,and 
3. install this module,<code>python setup.py install</code>
4. (hack)Now there should be an new folder <code>build</code> created in the root dir. For some reason the <code>align</code> folder under <code>$(STR-TDSL_RootDir)/maskrcnn_benchmark/modeling/one_stage_head</code> is not copied. Therefore on my computer i have to do the following manual copying to <code>$(AnacondaRootDir)\envs\pytorch-1.0\Lib\site-packages\maskrcnn_benchmark-0.1-py3.7-win-amd64.egg\maskrcnn_benchmark\modeling\one_stage_head</code>

