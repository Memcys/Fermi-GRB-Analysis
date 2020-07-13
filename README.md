
# Fermi GRB Analysis

本项目试图复现文献 <cite>[Xu2018][1]</cite> 中的步骤。


项目文件说明
============
- `data`: 设计用于存放数据图表。具体可由 `pkg/config.py` 或运行 `setup.py` 时指定（参见[安装 `pkg` 包](#安装-pkg-包)）
- `demo`: 使用 `pkg` 的示例代码（**使用入口**）。使用时建议将 `.py` 格式转化为 `.ipynb` （参见 [Jupytext 使用说明](#Jupytext-使用说明)
    - `main.py`: 调用 `pkg/lat/analysis.py` 实现主要的数据分析和可视化
    - `query-LAT.py`: 调用 `pkg/lat/query.py` 下载 *Fermi* LAT 数据
- `pkg`: 具体实现的底层代码
    - `config`: 配置相关模块
        - 'path.py`: 用于指定统一的数据图表路径，包括 `ROOT`, `FITS`, `TABLE` 和 `IMAGE`
        - `__init__.py`: 使 `pkg` 成为 [namespace package](https://docs.python.org/3/tutorial/modules.html#packages) 的模块的 *必要*文件
    - `lat`: *Fermi* LAT 相关模块
        - `analysis.py`: 数据分析和可视化的具体实现
        - `query.py`: 用于下载 *Fermi* LAT 数据
        - `__init__.py`
- `setup.py`: 安装本地包 `pkg` 的设置文件

`data/fits` 中的数据可由 `demo/query-LAT.py` 下载保存。示例中的数据可以从 https://github.com/Memcys/LAT-GRB-data 或其 [release](https://github.com/Memcys/LAT-GRB-data/releases/) 下载。下载（并解压）后可以通过 `cp` 或 `ln` 放入本项目的 `data/fits` 文件夹中。


搭建代码环境
==========
搭建虚拟环境
----------
推荐使用 Python 虚拟环境。Python 虚拟环境简介和创建方式可参见 https://docs.python.org/3/library/venv.html. 例如，可如下创建（请使用 Python 3.8 或以上版本。**Windows 下 `/` 应改为 `\`**）：

```Python
python3 -m venv /path/to/new/virtual/environment	# 创建虚拟环境（一次性）
source /path/to/new/virtual/environment/bin/activate	# 启用虚拟环境
pip3 install -U pip					# 更新 pip
pip3 install -r /path/to/requirements.txt		# 安装依赖包
```
如果下载速度慢，可更换下载镜像，比如用清华源（参考 https://mirrors.tuna.tsinghua.edu.cn/help/pypi/）
```
pip install pip -U
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

当然，也可以使用 Conda 创建和管理虚拟环境。可参考 https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html.


### 安装 `pkg` 包
在 `setup.py` 同级目录 (`ROOT`) 下，命令行运行
```
python setup.py install
```
运行后首先显示 4 个路径：
- `ROOT` （绝对路径）用于指定根目录路径，即 `Fermi-GBM-Analysis` 项目文件夹的绝对路径
- `FITS` （相对于 `ROOT` 的路径）用于存放 [FITS](https://fits.gsfc.nasa.gov/fits_standard.html) 数据文件
- `TABLE` （相对于 `ROOT` 的路径）用于存放中间数据表或下载的数据表
- `IMAGE` （相对于 `ROOT` 的路径）用于存放绘图

以及当前工作路径。

另外，也可以手动修改 `pkg/config/path.py` 后安装或更新（方法同安装） `pkg` 包。

此后可以在 Python 代码中通过 `from pkg import config` 等类似语法导入该包的模块。也可通过
```
pip uninstall pkg
```
卸载该包。


### 查看可更新的包
通过
```python
pip list --outdated
```
可以查看可更新的包。

[一条 issue](https://github.com/pypa/pip/issues/3819)中展示了更新所有可更新包的多种可能的命令。

**注意**：除非使用**虚拟环境**，否则不建议更新所有包，尤其是在 *nix 系统中。


浏览编辑代码文件
==============
对于 `.ipynb`，可使用 Jupyter Lab 打开文件夹或文件
```
jupyter lab /path/to/wherever/you/like
```
这将打开一个基于浏览器页面的 Jupyter Lab 环境。

使用其他编辑器/IDE 的方法不赘述。


***
## 待办
- [ ] 添加必要的 [docstring](https://numpydoc.readthedocs.io/en/latest/format.html)
- [ ] 使用 sphinx 构建程序说明文档
- [ ] 程序有待中使用的部分中间数据文件的生成方式已被抹去，待补充


参考文献
=======
[1]: Xu, H., & Ma, B.-Q. (2018). Regularity of high energy photon events from gamma ray bursts. Journal of Cosmology and Astroparticle Physics, 2018(01), 050–050. https://doi.org/10.1088/1475-7516/2018/01/050, http://arxiv.org/abs/1801.08084


附录
====

Jupytext 使用说明
----------------
更详细的使用说明请参见 [jupytext documentation](https://jupytext.readthedocs.io/en/latest/).
- `.py` 是 Python 程序文件
- `.ipynb` 是 IPython Notebook 文件，推荐用 [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/) 或 Jupyter Notebook 打开
- `.py` 和 `.ipynb` 文件可以通过 [jupytext](https://jupytext.readthedocs.io/en/latest/introduction.html) 同步
- 我本地使用 `demos` 中的 `.ipynb` 文件。为便于版本控制，仅上传了同步为 `.py` 的文件。

同步示例（把 `query-LAT.py` 同步为 `.ipynb`）：

```
jupytext --set-formats py,ipynb --sync query-LAT.py -o query-LAT.ipynb
```
反之，也可把 `.ipynb` 同步为 `.py`.