这个文件夹里包含 ray 适配 mlu 所需的内容. 其 diff 同时放在了 `diff.patch` 里.

原始适配基于官方 commit: 457d6e930e2d87354c9462b150be26a592508ea1, 其对应的 wheel 包在:
`https://s3-us-west-2.amazonaws.com/ray-wheels/master/457d6e930e2d87354c9462b150be26a592508ea1/ray-3.0.0.dev0-cp310-cp310-manylinux2014_x86_64.whl`

安装 ray 的 mlu 适配的步骤为:
1. 安装官方commit的wheel包:
   ```bash
   pip install https://s3-us-west-2.amazonaws.com/ray-wheels/master/457d6e930e2d87354c9462b150be26a592508ea1/ray-3.0.0.dev0-cp310-cp310-manylinux2014_x86_64.whl
   ```
2. 把 `__init__.py` 和 `mlu.py` 拷到包安装的地方. 其中, `__init__.py` 会把已有的 `__init__.py` 覆盖, `mlu.py` 会是一个全新的文件. 比如这样:
   ```bash
   cp ./__init__.py /usr/local/lib/python3.10/site-packages/ray/_private/accelerators
   cp ./mlu.py /usr/local/lib/python3.10/site-packages/ray/_private/accelerators
   cp ./nsight.py /usr/local/lib/python3.10/site-packages/ray/_private/runtime_env
   ```