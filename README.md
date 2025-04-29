# **Robot Learning: Training for Robotic Tasks**

This project explores imitation learning techniques using **Robomimic** and **Robosuite** to train robotic agents for object manipulation tasks. It implements Equivariant BC-RNN, a model that integrates SO(2) rotational symmetry into a recurrent framework to improve generalization in planar manipulation tasks. The repository includes preprocessing scripts, training scripts, and evaluation tools.

---

## **Author**
**Ekeno Lokwakai**

---

## **System Requirements**

### **Environment**
- **Operating System**: Tested on Ubuntu 20.04 (compatible with Ubuntu 22.04)
- **Python Version**: 3.8 or higher
- **GPU**: Optional, CUDA-enabled GPU for training acceleration (tested with CUDA 11.8)

### **Required Software & Libraries**
- **Robomimic**: [GitHub Repo](https://github.com/ARISE-Initiative/robomimic) (commit: `v0.3`, tested with this version)
- **Robosuite**: [GitHub Repo](https://github.com/ARISE-Initiative/robosuite) (commit: `v1.4`, tested with this version)
- **MuJoCo**: Physics engine for simulation (version 2.3.7)
- **PyTorch**: Neural network framework (version 2.0.1, with `torch`, `torchvision`, `torchaudio`)
- **Additional Packages**:
  ```bash
  pip install numpy==1.24.3 h5py==3.9.0 matplotlib==3.7.2 pandas==2.0.3 tqdm==4.66.1 tensorboard==2.14.0 imageio==2.31.1 opencv-python==4.8.0.76
  ```

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/EkenoSamson/Learning-Robots
   cd Learning-Robots
   ```
2. Install Robosuite and Robomimic dependencies:
   - Follow instructions in their respective GitHub repositories to set up Robosuite and Robomimic.
   - Ensure MuJoCo is installed and configured (see Robosuite’s documentation for setup).
3. Install additional Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   (If `requirements.txt` is not present, use the `pip install` command above to install dependencies.)

---

## **How to Run the Experiments**

### **Step 1: Preprocess the Dataset for SO(2) Equivariance**
Before training, preprocess the dataset to enforce SO(2) equivariance using the provided script. This step applies random 2D rotations to the x-y components of observations, ensuring the dataset is suitable for training an equivariant model.

1. Ensure the dataset (`low_dim_v15.hdf5`) is available in `~/robomimic/datasets/lift/ph/`. If not, download it from Robomimic’s dataset repository.
2. Run the preprocessing script:
   ```bash
   python equi_bc_rnn/make_SO2_equiv_dataset.py
   ```
   - **Input**: `~/robomimic/datasets/lift/ph/low_dim_v15.hdf5`
   - **Output**: A new dataset at `~/robomimic/datasets/lift/ph/low_dim_v15_so2.hdf5`
   - The script applies random SO(2) rotations to each demonstration’s observations (`robot0_eef_pos`, `robot0_eef_vel_lin`, `robot0_eef_vel_ang`).

### **Step 2: Inspect the Preprocessed Dataset**
Verify the dataset structure and contents:
```bash
python robomimic/robomimic/scripts/get_dataset_info.py --dataset ~/robomimic/datasets/lift/ph/low_dim_v15_so2.hdf5
```

### **Step 3: Train the Model**
Train the Equivariant BC-RNN model using the preprocessed dataset:
```bash
python robomimic/robomimic/scripts/train.py --config equi_bc_rnn/configs/equiv_bc_rnn_config.json
```
- Ensure `equiv_bc_rnn/configs/equiv_bc_rnn_config.json` is configured to use `~/robomimic/datasets/lift/ph/low_dim_v15_so2.hdf5` as the dataset.
- Training logs and checkpoints will be saved to `trained-rnn/`.

### **Step 4: View Training Logs**
Monitor training progress using TensorBoard:
```bash
tensorboard --logdir trained-rnn/ --bind_all
```
Access the TensorBoard interface at `http://localhost:6006` (or the specified port).

### **Step 5: Evaluate the Model**
Evaluate the trained model by running rollouts:
```bash
python robomimic/robomimic/scripts/run_trained_agent.py \
    --agent trained-rnn/model.pth \
    --n_rollouts 50 \
    --horizon 400 \
    --seed 0 \
    --video_path results/rollout_video.mp4
```
- This generates 50 rollouts, each with a horizon of 400 steps, and saves a video to `results/rollout_video.mp4`.

---

## **Reports**
The project report (IEEE format) is available in the `Documentation/` directory. It details the methodology, experiments, and results of the Equivariant BC-RNN model.

## **Experiment Videos**
- [Lift Experiment Video](https://youtube.com/shorts/ccmSQhh74X8?feature=share)

---

## **Notes**
- Ensure the Robomimic and Robosuite repositories are cloned and set up as per their documentation. This project does not include their code to avoid plagiarism, per the instructor’s guidelines.
- The `square_experiment/` directory has been removed as it was unused, in compliance with the deliverable requirements.
- All scripts assume the dataset paths are correctly set. Adjust paths in scripts or configs if your dataset is located elsewhere.