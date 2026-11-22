# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

# Local patch (2026-11-22): praxis targets torch >= 2.15 APIs (FlexKernelOptions and a
# 2.15-only inductor compile option). On older torch builds (e.g. 2.8.0+cu126 on GTX
# 1080 Ti) the imports fail at module level. Make the package import-tolerant so other
# policies (ACT et al.) can load through the shared policy registry; `--policy.type=praxis`
# simply isn't available on those builds.
try:
    from .configuration_praxis import PraxisConfig
    from .modeling_praxis import PraxisPolicy
    from .processor_praxis import make_praxis_pre_post_processors

    __all__ = ["PraxisConfig", "PraxisPolicy", "make_praxis_pre_post_processors"]
except (ImportError, RuntimeError) as _praxis_import_error:  # noqa: F841
    __all__: list[str] = []
