from utils.buffer import CompressedTrajectoryBuffer
from utils.normalizer import LinearNormalizer, NestedDictLinearNormalizer
from utils.obs_utils import unflatten_obs
from utils.sampler import TrajectorySampler

from generative_policies import InverseDynamicsModel

def main(buffer_path, buffer_metadata):
    buffer: CompressedTrajectoryBuffer = CompressedTrajectoryBuffer(buffer_path, buffer_metadata)
    
    id_model: InverseDynamicsModel = InverseDynamicsModel()

    relabel_trajectory_dataset_with_id(buffer)

def relabel_trajectory_dataset_with_id(id_model: InverseDynamicsModel, buffer: CompressedTrajectoryBuffer) -> TrajectoryDataset:
    """
    1. Loads the trajectory dataset
    2. For each (state,next_state), computes the inverse dynamics according to id_model.sample(state,next_state)
    """
    pass