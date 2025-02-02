# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .numpy_dataset import load_numpy_dataset, load_numpy_dataset_shifted_split
from .toy_ours import generate_toy_ours_dataset
from .toy_vargrad import generate_vargrad_dataset
from .uci import UCI_DATASETS_PATH, UCIDataset
