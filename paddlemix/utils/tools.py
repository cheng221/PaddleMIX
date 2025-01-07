# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle


def compare_version(version, pair_version):
    """
    Args:
        version (str): The first version string needed to be compared.
            The format of version string should be as follow : "xxx.yyy.zzz".
        pair_version (str): The second version string needed to be compared.
             The format of version string should be as follow : "xxx.yyy.zzz".
    Returns:
        int: The result of comparison. 1 means version > pair_version; 0 means
            version = pair_version; -1 means version < pair_version.

    Examples:
        >>> compare_version("2.2.1", "2.2.0")
        >>> 1
        >>> compare_version("2.2.0", "2.2.0")
        >>> 0
        >>> compare_version("2.2.0-rc0", "2.2.0")
        >>> -1
        >>> compare_version("2.3.0-rc0", "2.2.0")
        >>> 1
    """
    version = version.strip()
    pair_version = pair_version.strip()
    if version == pair_version:
        return 0
    version_list = version.split(".")
    pair_version_list = pair_version.split(".")
    for version_code, pair_version_code in zip(version_list, pair_version_list):
        if not version_code.isnumeric():
            return -1
        if not pair_version_code.isnumeric():
            return 1
        if int(version_code) > int(pair_version_code):
            return 1
        elif int(version_code) < int(pair_version_code):
            return -1
    return 0


def get_env_device():
    """
    Return the device name of running environment.
    """
    if paddle.is_compiled_with_cuda():
        return "gpu"
    elif "npu" in paddle.device.get_all_custom_device_type():
        return "npu"
    elif paddle.is_compiled_with_rocm():
        return "rocm"
    elif paddle.is_compiled_with_xpu():
        return "xpu"
    return "cpu"


def check_dtype_compatibility():
    """
    检查当前环境下可用的数据类型
    返回最优的可用数据类型
    """
    if not paddle.is_compiled_with_cuda():
        print("CUDA not available, falling back to float32")
        return paddle.float32

    # 获取GPU计算能力
    gpu_arch = paddle.device.cuda.get_device_capability()
    if gpu_arch is None:
        print("Unable to determine GPU architecture, falling back to float32")
        return paddle.float32

    major, minor = gpu_arch
    compute_capability = major + minor / 10
    print(f"GPU compute capability: {compute_capability}")

    try:
        # 测试bfloat16兼容性
        if compute_capability >= 8.0:  # Ampere及更新架构
            test_tensor = paddle.zeros([2, 2], dtype="bfloat16")
            paddle.matmul(test_tensor, test_tensor)
            print("bfloat16 is supported and working")
            return paddle.bfloat16
    except Exception as e:
        print(f"bfloat16 test failed: {str(e)}")

    try:
        # 测试float16兼容性
        if compute_capability >= 5.3:  # Maxwell及更新架构
            test_tensor = paddle.zeros([2, 2], dtype="float16")
            paddle.matmul(test_tensor, test_tensor)
            print("float16 is supported and working")
            return paddle.float16
    except Exception as e:
        print(f"float16 test failed: {str(e)}")

    print("Falling back to float32 due to compatibility issues")
    return paddle.float32
