if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
os.environ['MUJOCO_GL'] = 'egl'
import cv2
import hydra
import collections
import numpy as np
from tqdm.auto import tqdm
from training.mavis.mavis_incremental_bc import MAVISIncrementalBehavioralCloning
from mavis_mujoco_gym.utils.create_env import create_env

class MAVISHumanGatedDataAggregation(MAVISIncrementalBehavioralCloning):
    def __init__(self, cfg):
        super().__init__(cfg)

    def collect_one_rollout(self, interactive=False):
        self.dataset.create_new_rollout_dataset()

        data_collection_env = create_env(env_id=self.cfg.env_id,
                                         env_configs=self.cfg.env_configs)
        obs, info = data_collection_env.reset()

        obs['rgb'] = cv2.resize(obs['rgb'], (self.cfg.downsampled_img_width, self.cfg.downsampled_img_height))
        obs_deque = collections.deque([obs] * self.cfg.network.obs_horizon, maxlen=self.cfg.network.obs_horizon)

        demo_step_idx = 0

        done = False
        terminated = False
        truncated = False

        with tqdm(total=self.cfg.max_num_steps_per_rollout, desc="Number of Demonstration Steps") as pbar:
            while demo_step_idx < self.cfg.max_num_steps_per_rollout and not done:
                images = np.stack([x['rgb'] for x in obs_deque])
                agent_poses = np.stack([x['state'] for x in obs_deque])

                expert_action = self.expert.predict_action((images, agent_poses))
                learner_action = self.learner.predict_action((images, agent_poses))

                for i in range(len(expert_action)):
                    if interactive:
                        curr_expert_action = expert_action[i]
                        curr_learner_action = learner_action[i]

                        cos_sim = np.dot(curr_expert_action, curr_learner_action) / (np.linalg.norm(curr_expert_action) * np.linalg.norm(curr_learner_action))

                        if cos_sim < self.cfg.expert_overtake_cos_sim_threshold:
                            curr_action = curr_expert_action
                            is_expert_label = True
                            self.curr_expert_label_num += 1
                        else:
                            curr_action = curr_learner_action
                            is_expert_label = False
                    else:
                        curr_action = expert_action[i]
                        is_expert_label = True
                        self.curr_expert_label_num += 1

                    self.dataset.append_to_curr_rollout_dataset(rgb_img=obs['rgb'],
                                                                state=obs['state'],
                                                                action=curr_action,
                                                                is_expert_label=is_expert_label)


                    obs, reward, terminated, truncated, info = data_collection_env.step(curr_action)

                    obs['rgb'] = cv2.resize(obs["rgb"], (self.cfg.downsampled_img_width, self.cfg.downsampled_img_height))

                    done = terminated or truncated
                    obs_deque.append(obs)
                    demo_step_idx += 1
                    pbar.update(1)

                    if demo_step_idx > self.cfg.max_num_steps_per_rollout or truncated:
                        print("Current demonstration rollout exceeds maximum number of steps.")
                        self.dataset.remove_curr_rollout_dataset()
                        data_collection_env.close()
                        del data_collection_env
                        return False
                    if terminated:
                        self.dataset.stop_recording_curr_rollout_dataset_and_add_it_to_this_object()
                        data_collection_env.close()
                        del data_collection_env
                        # print("incremental_num_expert_labels_from_each_rollout: ", self.dataset.incremental_num_expert_labels_from_each_rollout)
                        return True

@hydra.main(
    version_base=None,
    config_path='../../config',
    config_name="mavis_pick_and_place_hg_dagger_dp_unet")
def main(cfg):
    mavis_hg_dagger = MAVISHumanGatedDataAggregation(cfg)
    mavis_hg_dagger.run()

if __name__ == "__main__":
    main()
