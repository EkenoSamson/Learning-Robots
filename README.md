# **Robot Learning: Training for Robotic Tasks**

This project explores imitation learning techniques using **Robomimic** and **Robosuite** to train robotic agents for object manipulation tasks. The experiments are conducted on different datasets using different models and the learning performances are evaluated under varying conditions.

---

## **Author**
**Ekeno Lokwakai**

---

## **System Requirements**

### **Environment**
- **Operating System**: Ubuntu 20.04 / 22.04  
- **Python Version**: 3.8+  
- **GPU**: (Optional) CUDA-enabled GPU for training acceleration  

### **Required Software & Libraries**
- **Robomimic** ([GitHub Repo](https://github.com/ARISE-Initiative/robomimic))  
- **Robosuite** ([GitHub Repo](https://github.com/ARISE-Initiative/robosuite))  
- **MuJoCo**: Physics engine for simulation  
- **PyTorch**: Neural network framework (`torch`, `torchvision`, `torchaudio`)  
- **Additional Packages**:  
  ```bash
  pip install numpy h5py matplotlib pandas tqdm tensorboard imageio opencv-python

### **Installation**
1. Clone the repository
```
    git clone https://github.com/EkenoSamson/Learning-Robots
    cd Learning-Robots
```

### **How to run the experiments**
i. inspecting a dataset
```
    python robomimic/robomimic/scripts/get_dataset_info.py --dataset dataset/ph/dataset.hdf5
```


ii. running the training
```
    python robomimic/robomimic/scripts/train.py --config <experiment>/configs/<config_file>.json
```

iii. Viewing the training logs
```
    tensorboard --logdir trained-rnn/ --bind_all
```

iv. Evaluating the model
```
    python robomimic/robomimic/scripts/run_trained_agent.py \
    --agent path/to/model.pth \
    --n_rollouts 50 \
    --horizon 400 \
    --seed 0 \
    --video_path results/rollout_video.mp4
````

### **Reports**
Access the report in the ```Documentation``` directory.


### **Experiements Videos**
+ [Lift experiment](https://youtube.com/shorts/ccmSQhh74X8?feature=share)



