# mixtures-of-depth-of-mixtures


# other implementations

- https://github.com/astramind-ai/Mixture-of-depths/tree/main
  - roughly started from the implementation here:
  - that one does not work with many models/does single sample so rewrote the forward, also device_map='auto' will break the model due to the hooks on the model forwards
  - another change is allowing skip_position_ids to be passed in as older llama based model files will break with this as is
- https://github.com/kyegomez/Mixture-of-Depths/blob/main/mixture_of_depths/main.py
  - this one seems maybe like it is good since it uses gather/scatter but has some errors/issues
- https://github.com/Mixture-AI/Mixture-of-Depths/blob/master/MoD/MoD.py

